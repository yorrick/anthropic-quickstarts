"""
Entrypoint for streamlit, see https://docs.streamlit.io/
"""

import asyncio
import base64
import logging
import os
import pickle
import subprocess
import traceback
from datetime import datetime, timedelta
from enum import StrEnum
from functools import partial
from pathlib import PosixPath
from typing import cast

import httpx
import streamlit as st
from anthropic import RateLimitError
from anthropic.types.beta import (
    BetaContentBlockParam,
    BetaTextBlockParam,
)
from streamlit.delta_generator import DeltaGenerator

from computer_use_demo.loop import (
    PROVIDER_TO_DEFAULT_MODEL_NAME,
    APIProvider,
    sampling_loop,
)
from computer_use_demo.tools import ToolResult

# Setup logging configuration
CONFIG_DIR = PosixPath("~/.anthropic").expanduser()
CONFIG_DIR.mkdir(parents=True, exist_ok=True)
log_file = CONFIG_DIR / "streamlit_debug.log"

# Create logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Create handlers
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

# Create formatter and add it to handlers
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Add handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# Set higher level for other loggers to suppress their output
logging.getLogger("watchdog").setLevel(logging.WARNING)
logging.getLogger("asyncio").setLevel(logging.WARNING)
logging.getLogger("streamlit").setLevel(logging.WARNING)

# Clear any existing handlers from the root logger
logging.getLogger().handlers.clear()

API_KEY_FILE = CONFIG_DIR / "api_key"

STREAMLIT_STYLE = """
<style>
    /* Hide chat input while agent loop is running */
    .stApp[data-teststate=running] .stChatInput textarea,
    .stApp[data-test-script-state=running] .stChatInput textarea {
        display: none;
    }
     /* Hide the streamlit deploy button */
    .stAppDeployButton {
        visibility: hidden;
    }
</style>
"""

WARNING_TEXT = "⚠️ Security Alert: Never provide access to sensitive accounts or data, as malicious web content can hijack Claude's behavior"


class Sender(StrEnum):
    USER = "user"
    BOT = "assistant"
    TOOL = "tool"


class PersistentDict:
    def __init__(self, filename: str):
        self.filename = CONFIG_DIR / filename
        self.cache = self._load()

    def _load(self):
        try:
            if self.filename.exists():
                with open(self.filename, "rb") as f:
                    return pickle.load(f)
        except Exception as e:
            logger.error(f"Failed to load state: {e}")
        return {}

    def _save(self):
        try:
            with open(self.filename, "wb") as f:
                pickle.dump(self.cache, f)
            self.filename.chmod(0o600)
        except Exception as e:
            logger.error(f"Failed to save state: {e}")

    def get(self, key, default=None):
        return self.cache.get(key, default)

    def set(self, key, value):
        self.cache[key] = value
        self._save()

    def clear(self):
        self.cache.clear()
        self._save()

    def clear_conversation(self):
        """Clear only conversation-related state"""
        keys_to_clear = ["messages", "tools", "responses"]
        for key in keys_to_clear:
            if key in self.cache:
                del self.cache[key]
        self._save()


# Create global state manager
state_manager = PersistentDict("streamlit_state.pkl")


def setup_state():
    """Initialize and manage application state"""
    logger.debug("Setting up state")

    # Load persisted state
    messages = state_manager.get("messages", [])
    tools = state_manager.get("tools", {})
    responses = state_manager.get("responses", {})

    logger.debug(f"Loaded persisted state with {len(messages)} messages")

    # Initialize default state
    defaults = {
        "messages": messages,
        "api_key": load_from_storage("api_key") or os.getenv("ANTHROPIC_API_KEY", ""),
        "provider": os.getenv("API_PROVIDER", "anthropic") or APIProvider.ANTHROPIC,
        "provider_radio": st.session_state.get(
            "provider", os.getenv("API_PROVIDER", "anthropic") or APIProvider.ANTHROPIC
        ),
        "auth_validated": False,
        "responses": responses,
        "tools": tools,
        "only_n_most_recent_images": 10,
        "custom_system_prompt": load_from_storage("system_prompt") or "",
        "hide_images": False,
        "session_id": datetime.now().isoformat(),
    }

    # Initialize session state with defaults
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    # Ensure model is set correctly
    if "model" not in st.session_state:
        _reset_model()


def reset_conversation_only():
    """Reset only the conversation state without affecting X11 or other settings"""
    logger.debug("Resetting conversation only")
    # Clear conversation-related state
    state_manager.clear_conversation()

    # Reset only conversation-related session state
    st.session_state.messages = []
    st.session_state.tools = {}
    st.session_state.responses = {}

    # Keep the session ID to maintain other settings
    current_session_id = st.session_state.session_id

    # Update the session state with empty conversation
    setup_state()

    # Restore the session ID
    st.session_state.session_id = current_session_id

    logger.debug("Conversation reset complete")


def update_persisted_state():
    """Update the persisted state with current session values"""
    logger.debug("Updating persisted state")
    try:
        state_manager.set("messages", st.session_state.messages)
        state_manager.set("tools", st.session_state.tools)
        state_manager.set("responses", st.session_state.responses)

        logger.debug(
            f"Successfully saved state with {len(st.session_state.messages)} messages"
        )
    except Exception as e:
        logger.error(f"Failed to update persisted state: {e}")


def _reset_model():
    """Reset the model based on the current provider"""
    logger.debug("Resetting model")
    st.session_state.model = PROVIDER_TO_DEFAULT_MODEL_NAME[
        cast(APIProvider, st.session_state.provider)
    ]
    logger.debug(f"Model set to: {st.session_state.model}")


def validate_auth(provider: APIProvider, api_key: str | None):
    """Validate authentication credentials for the chosen provider"""
    if provider == APIProvider.ANTHROPIC:
        if not api_key:
            return "Enter your Anthropic API key in the sidebar to continue."
    if provider == APIProvider.BEDROCK:
        import boto3

        if not boto3.Session().get_credentials():
            return "You must have AWS credentials set up to use the Bedrock API."
    if provider == APIProvider.VERTEX:
        import google.auth
        from google.auth.exceptions import DefaultCredentialsError

        if not os.environ.get("CLOUD_ML_REGION"):
            return "Set the CLOUD_ML_REGION environment variable to use the Vertex API."
        try:
            google.auth.default(
                scopes=["https://www.googleapis.com/auth/cloud-platform"],
            )
        except DefaultCredentialsError:
            return "Your google cloud credentials are not set up correctly."


def load_from_storage(filename: str) -> str | None:
    """Load data from a file in the storage directory."""
    try:
        file_path = CONFIG_DIR / filename
        if file_path.exists():
            data = file_path.read_text().strip()
            if data:
                return data
    except Exception as e:
        logger.error(f"Error loading {filename}: {e}")
    return None


def save_to_storage(filename: str, data: str) -> None:
    """Save data to a file in the storage directory."""
    try:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        file_path = CONFIG_DIR / filename
        file_path.write_text(data)
        # Ensure only user can read/write the file
        file_path.chmod(0o600)
    except Exception as e:
        logger.error(f"Error saving {filename}: {e}")


def _api_response_callback(
    request: httpx.Request,
    response: httpx.Response | object | None,
    error: Exception | None,
    tab: DeltaGenerator,
    response_state: dict[str, tuple[httpx.Request, httpx.Response | object | None]],
):
    """Handle an API response by storing it to state and rendering it."""
    response_id = datetime.now().isoformat()
    response_state[response_id] = (request, response)
    if error:
        _render_error(error)
    _render_api_response(request, response, response_id, tab)
    update_persisted_state()


def _tool_output_callback(
    tool_output: ToolResult, tool_id: str, tool_state: dict[str, ToolResult]
):
    """Handle a tool output by storing it to state and rendering it."""
    tool_state[tool_id] = tool_output
    _render_message(Sender.TOOL, tool_output)
    update_persisted_state()


def _render_api_response(
    request: httpx.Request,
    response: httpx.Response | object | None,
    response_id: str,
    tab: DeltaGenerator,
):
    """Render an API response to a streamlit tab"""
    with tab:
        with st.expander(f"Request/Response ({response_id})"):
            newline = "\n\n"
            st.markdown(
                f"`{request.method} {request.url}`{newline}{newline.join(f'`{k}: {v}`' for k, v in request.headers.items())}"
            )
            st.json(request.read().decode())
            st.markdown("---")
            if isinstance(response, httpx.Response):
                st.markdown(
                    f"`{response.status_code}`{newline}{newline.join(f'`{k}: {v}`' for k, v in response.headers.items())}"
                )
                st.json(response.text)
            else:
                st.write(response)


def _render_error(error: Exception):
    """Render an error message with appropriate formatting"""
    if isinstance(error, RateLimitError):
        body = "You have been rate limited."
        if retry_after := error.response.headers.get("retry-after"):
            body += f" **Retry after {str(timedelta(seconds=int(retry_after)))} (HH:MM:SS).** See our API [documentation](https://docs.anthropic.com/en/api/rate-limits) for more details."
        body += f"\n\n{error.message}"
    else:
        body = str(error)
        body += "\n\n**Traceback:**"
        lines = "\n".join(traceback.format_exception(error))
        body += f"\n\n```{lines}```"
    save_to_storage(f"error_{datetime.now().timestamp()}.md", body)
    st.error(f"**{error.__class__.__name__}**\n\n{body}", icon=":material/error:")


def _render_message(
    sender: Sender,
    message: str | BetaContentBlockParam | ToolResult,
):
    """Convert input from the user or output from the agent to a streamlit message."""
    # streamlit's hotreloading breaks isinstance checks, so we need to check for class names
    is_tool_result = not isinstance(message, str | dict)
    if not message or (
        is_tool_result
        and st.session_state.hide_images
        and not hasattr(message, "error")
        and not hasattr(message, "output")
    ):
        return
    with st.chat_message(sender):
        if is_tool_result:
            message = cast(ToolResult, message)
            if message.output:
                if message.__class__.__name__ == "CLIResult":
                    st.code(message.output)
                else:
                    st.markdown(message.output)
            if message.error:
                st.error(message.error)
            if message.base64_image and not st.session_state.hide_images:
                st.image(base64.b64decode(message.base64_image))
        elif isinstance(message, dict):
            if message["type"] == "text":
                st.write(message["text"])
            elif message["type"] == "tool_use":
                st.code(f'Tool Use: {message["name"]}\nInput: {message["input"]}')
            else:
                # only expected return types are text and tool_use
                raise Exception(f'Unexpected response type {message["type"]}')
        else:
            st.markdown(message)


async def main():
    """Render loop for streamlit"""
    logger.debug("Starting main loop")
    logger.debug(f"Script run ID: {st.session_state.get('session_id', 'NOT_SET')}")

    setup_state()

    st.markdown(STREAMLIT_STYLE, unsafe_allow_html=True)

    st.title("Claude Computer Use Demo")

    if not os.getenv("HIDE_WARNING", False):
        st.warning(WARNING_TEXT)

    with st.sidebar:

        def _reset_api_provider():
            logger.debug("Resetting API provider")
            if st.session_state.provider_radio != st.session_state.provider:
                _reset_model()
                st.session_state.provider = st.session_state.provider_radio
                st.session_state.auth_validated = False
                update_persisted_state()

        provider_options = [option.value for option in APIProvider]
        st.radio(
            "API Provider",
            options=provider_options,
            key="provider_radio",
            format_func=lambda x: x.title(),
            on_change=_reset_api_provider,
        )

        st.text_input("Model", key="model")

        if st.session_state.provider == APIProvider.ANTHROPIC:
            st.text_input(
                "Anthropic API Key",
                type="password",
                key="api_key",
                on_change=lambda: save_to_storage("api_key", st.session_state.api_key),
            )

        st.number_input(
            "Only send N most recent images",
            min_value=0,
            key="only_n_most_recent_images",
            help="To decrease the total tokens sent, remove older screenshots from the conversation",
        )
        st.text_area(
            "Custom System Prompt Suffix",
            key="custom_system_prompt",
            help="Additional instructions to append to the system prompt. see computer_use_demo/loop.py for the base system prompt.",
            on_change=lambda: save_to_storage(
                "system_prompt", st.session_state.custom_system_prompt
            ),
        )
        st.checkbox("Hide screenshots", key="hide_images")

        # Add two columns for reset buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Reset All", type="primary"):
                logger.debug("Reset all button clicked")
                with st.spinner("Resetting..."):
                    state_manager.clear()
                    st.session_state.clear()
                    setup_state()

                    subprocess.run("pkill Xvfb; pkill tint2", shell=True)  # noqa: ASYNC221
                    await asyncio.sleep(1)
                    subprocess.run("./start_all.sh", shell=True)  # noqa: ASYNC221

        with col2:
            if st.button("Reset Chat Only", type="secondary"):
                logger.debug("Reset conversation only button clicked")
                with st.spinner("Resetting conversation..."):
                    reset_conversation_only()

    if not st.session_state.auth_validated:
        if auth_error := validate_auth(
            st.session_state.provider, st.session_state.api_key
        ):
            st.warning(f"Please resolve the following auth issue:\n\n{auth_error}")
            return
        else:
            st.session_state.auth_validated = True
            update_persisted_state()

    chat, http_logs = st.tabs(["Chat", "HTTP Exchange Logs"])
    new_message = st.chat_input(
        "Type a message to send to Claude to control the computer..."
    )

    with chat:
        logger.debug("Rendering chat messages")
        # render past chats
        for message in st.session_state.messages:
            if isinstance(message["content"], str):
                _render_message(message["role"], message["content"])
            elif isinstance(message["content"], list):
                for block in message["content"]:
                    if isinstance(block, dict) and block["type"] == "tool_result":
                        _render_message(
                            Sender.TOOL, st.session_state.tools[block["tool_use_id"]]
                        )
                    else:
                        _render_message(
                            message["role"],
                            cast(BetaContentBlockParam | ToolResult, block),
                        )

        # render past http exchanges
        for identity, (request, response) in st.session_state.responses.items():
            _render_api_response(request, response, identity, http_logs)

        # handle new message
        if new_message:
            logger.debug(f"New message received: {new_message[:50]}...")
            st.session_state.messages.append(
                {
                    "role": Sender.USER,
                    "content": [BetaTextBlockParam(type="text", text=new_message)],
                }
            )
            _render_message(Sender.USER, new_message)
            update_persisted_state()
            logger.debug(
                f"Message count after append: {len(st.session_state.messages)}"
            )

        try:
            most_recent_message = st.session_state["messages"][-1]
        except IndexError:
            logger.debug("No messages in state")
            return

        if most_recent_message["role"] is not Sender.USER:
            logger.debug("Most recent message is not from user")
            return

        with st.spinner("Running Agent..."):
            logger.debug("Starting agent sampling loop")
            st.session_state.messages = await sampling_loop(
                system_prompt_suffix=st.session_state.custom_system_prompt,
                model=st.session_state.model,
                provider=st.session_state.provider,
                messages=st.session_state.messages,
                output_callback=partial(_render_message, Sender.BOT),
                tool_output_callback=partial(
                    _tool_output_callback, tool_state=st.session_state.tools
                ),
                api_response_callback=partial(
                    _api_response_callback,
                    tab=http_logs,
                    response_state=st.session_state.responses,
                ),
                api_key=st.session_state.api_key,
                only_n_most_recent_images=st.session_state.only_n_most_recent_images,
            )
            logger.debug("Agent loop completed")
            update_persisted_state()
            logger.debug(f"Final message count: {len(st.session_state.messages)}")


if __name__ == "__main__":
    logger.debug("Application starting")
    asyncio.run(main())
