import asyncio
import os
from typing import Optional
from dotenv import load_dotenv
from azure.ai.projects.aio import AIProjectClient
from azure.identity.aio import DefaultAzureCredential
from azure.ai.projects.models import Agent, AgentThread, AsyncFunctionTool, AsyncToolSet

from utils.stream_handler import StreamEventHandler
from utils.utilities import Utilities
from functions.user_functions import fetch_current_datetime

# Configuration constants
AGENT_NAME = "test_agent"
MAX_COMPLETION_TOKENS = 10240
MAX_PROMPT_TOKENS = 20480
TEMPERATURE = 0.1
TOP_P = 0.1

# File containing instructions for the agent
INSTRUCTIONS_FILE = "function_calling.txt"


# --- Environment and Credential Managers ---

class EnvironmentManager:
    @staticmethod
    def load_environment():
        load_dotenv()

    @staticmethod
    def get_required_env_var(name: str) -> str:
        value = os.getenv(name)
        if not value:
            raise ValueError(f"Missing required environment variable: {name}")
        return value


class AzureCredentialManager:
    def __init__(self):
        self.credential = DefaultAzureCredential()

    async def close(self):
        await self.credential.close()

    def get_credential(self):
        return self.credential


# --- Azure AI Client and Agent Setup ---   
class AIProjectClientManager:
    def __init__(self, connection_string: str, credential):
        self.project_client = AIProjectClient.from_connection_string(
            conn_str=connection_string,
            credential=credential
        )
        
        # Create a proper async toolset with the function
        self.toolset = AsyncToolSet()
        # Add the function wrapped in an AsyncFunctionTool
        self.toolset.add(AsyncFunctionTool([fetch_current_datetime]))  # ✅ Correct method

    async def close(self):
        await self.project_client.close()

    def get_client(self):
        return self.project_client
    
    def get_toolset(self):
        return self.toolset


class AgentManager:
    def __init__(self, project_client: AIProjectClient, toolset: AsyncToolSet):
        self.project_client = project_client
        self.toolset = toolset  # <-- Fix: Store the toolset

    async def get_or_create_agent(self, name: str, model: str, instructions: str) -> Agent:
        agents = await self.project_client.agents.list_agents()
        for agent in agents.data:
            if agent.name == name:
                print(f"Using existing agent: {agent.id}")
                return agent

        agent = await self.project_client.agents.create_agent(
            name=name,
            model=model,
            instructions=instructions,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            toolset=self.toolset
        )
        print(f"Created new agent: {agent.id}")
        return agent


# --- Thread Manager with Stream Handling ---

class ThreadManager:
    def __init__(self, project_client: AIProjectClient, utilities: Utilities):
        self.project_client = project_client
        self.utilities = utilities

    async def create_thread(self) -> AgentThread:
        thread = await self.project_client.agents.create_thread()
        print(f"Created thread: {thread.id}")
        return thread

    async def post_message(self, thread: AgentThread, agent: Agent, message: str,
                           functions: Optional[AsyncToolSet] = None):  # <-- Fix: Changed type to AsyncToolSet
        await self.project_client.agents.create_message(
            thread_id=thread.id,
            role="user",
            content=message,
        )

        stream = await self.project_client.agents.create_stream(
            thread_id=thread.id,
            agent_id=agent.id,
            event_handler=StreamEventHandler(
                functions=functions,
                project_client=self.project_client,
                utilities=self.utilities,
            ),
            max_completion_tokens=MAX_COMPLETION_TOKENS,
            max_prompt_tokens=MAX_PROMPT_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            instructions=agent.instructions,
        )

        async with stream as s:
            await s.until_done()


# --- Chat Application ---

class ChatApplication:
    def __init__(self):
        EnvironmentManager.load_environment()
        self.project_connection_string = EnvironmentManager.get_required_env_var("PROJECT_CONNECTION_STRING")
        self.model_deployment_name = EnvironmentManager.get_required_env_var("MODEL_DEPLOYMENT_NAME")
        self.agent_name = os.getenv("AGENT_NAME", AGENT_NAME)
        self.credential_manager = AzureCredentialManager()
        self.utilities = Utilities()
        self.toolset = None  # <-- Fix: Renamed from functions to toolset
        self.agent = None
        self.thread = None

    async def initialize(self):
        credential = self.credential_manager.get_credential()
        self.client_manager = AIProjectClientManager(self.project_connection_string, credential)
        self.project_client = self.client_manager.get_client()
        self.toolset = self.client_manager.get_toolset()  # <-- Fix: Get the toolset, not a property
        self.agent_manager = AgentManager(self.project_client, self.toolset)  # <-- Fix: Pass the toolset
        self.thread_manager = ThreadManager(self.project_client, self.utilities)

        instructions = "You are a time-only assistant. Your ONLY purpose is to provide the current date and time in a user's requested city. You must NEVER provide any other information, explanations, or engage in any other conversation. If asked anything other than about time, respond with 'I can only provide the current date and time in a requested city.'"
        if INSTRUCTIONS_FILE:
            try:
                instructions = self.utilities.load_instructions(INSTRUCTIONS_FILE)
            except Exception as e:
                print(f"Warning: Could not load instructions file: {e}")

        self.agent = await self.agent_manager.get_or_create_agent(
            name=self.agent_name,
            model=self.model_deployment_name,
            instructions=instructions
        )

        self.thread = await self.thread_manager.create_thread()

    async def cleanup(self):
        await self.client_manager.close()
        await self.credential_manager.close()

    async def run(self):
        await self.initialize()
        print("\nChat started. Type 'exit' to quit.\n")

        while True:
            user_input = input("You: ").strip()
            if user_input.lower() in {"exit", "quit"}:
                break

            try:
                await self.thread_manager.post_message(
                    thread=self.thread,
                    agent=self.agent,
                    message=user_input,
                    functions=self.toolset  # <-- Fix: Pass the toolset
                )
            except Exception as e:
                print(f"[Error] {e}")

        print("Cleaning up resources...")
        await self.cleanup()


# --- Main Execution ---

async def main():
    app = ChatApplication()
    try:
        await app.run()
    finally:
        await app.cleanup()


if __name__ == "__main__":
    asyncio.run(main())