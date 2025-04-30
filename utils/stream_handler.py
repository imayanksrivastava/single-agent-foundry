from typing import Any
from azure.ai.projects.aio import AIProjectClient
from azure.ai.projects.models import (
    AsyncAgentEventHandler,
    AsyncFunctionTool,
    MessageDeltaChunk,
    MessageStatus,
    RunStep,
    RunStepDeltaChunk,
    ThreadMessage,
    ThreadRun,
)

from utils.utilities import Utilities


class StreamEventHandler(AsyncAgentEventHandler[str]):
    """Handles streaming events from Azure Agent and prints responses."""

    def __init__(self, functions: AsyncFunctionTool, project_client: AIProjectClient, utilities: Utilities) -> None:
        super().__init__()
        self.functions = functions
        self.project_client = project_client
        self.util = utilities

        self.is_printing_response = False  # Flag for response start
        self.response_buffer = ""          # Buffer for fallback or debugging

    def _print_agent_prefix(self):
        if not self.is_printing_response:
            print("Agent: ", end="", flush=True)
            self.is_printing_response = True

    async def on_message_delta(self, delta: MessageDeltaChunk) -> None:
        """Handle incoming streamed tokens from the agent."""
        if delta and hasattr(delta, 'text') and delta.text:
            self._print_agent_prefix()
            # self.util.log_token_blue(delta.text)  # This already prints and flushes
            self.response_buffer += delta.text

            try:
                self.util.log_token_blue(delta.text)
            except Exception:
                pass

    async def on_thread_message(self, message: ThreadMessage) -> None:
        """Handles full messages, including file attachments."""
        if not message:
            return

        try:
            if hasattr(message, 'file_ids') and message.file_ids:
                await self.util.get_files(message, self.project_client)
        except Exception as e:
            print(f"Error processing files: {str(e)}")

    async def on_thread_run(self, run: ThreadRun) -> None:
        """Prints failed run status."""
        try:
            if run and run.status == "failed":
                print(f"\nRun failed. Error: {run.last_error}")
        except Exception:
            pass

    async def on_run_step(self, step: RunStep) -> None:
        """(Optional) Handle intermediate run steps."""
        pass

    async def on_run_step_delta(self, delta: RunStepDeltaChunk) -> None:
        """(Optional) Handle streamed run steps."""
        pass

    async def on_error(self, data: str) -> None:
        """Handle error events and reset flags."""
        print(f"\nAn error occurred: {data}")
        self.is_printing_response = False
        self.response_buffer = ""

    async def on_done(self, data: Any = None) -> None:
        """Handle end of stream: newline + reset."""
        if self.is_printing_response:
            print()  # Finish the line
        self.is_printing_response = False
        self.response_buffer = ""

    async def on_unhandled_event(self, event_type: str, event_data: Any) -> None:
        """Debug any unexpected events."""
        print(f"\nUnhandled Event Type: {event_type}")