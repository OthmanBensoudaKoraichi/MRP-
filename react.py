from dotenv import load_dotenv
from langchain import hub
from langchain.agents import AgentExecutor, create_structured_chat_agent
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain.tools.base import StructuredTool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from langchain.tools import StructuredTool
import datetime
from wikipedia import summary
# Load environment variables from .env file
load_dotenv()





# Define Input Schemas for StructuredTool
class TimeInput(BaseModel):
    format: str | None = Field(default=None, description="Optional format for the current date and time. If not provided, returns the current time in a readable format.")

class WikipediaInput(BaseModel):
    query: str = Field(..., description="The topic to search on Wikipedia")

def get_current_time(format: str | None = None):
    """Returns the current time in a readable format or in the specified format if provided."""
    import datetime
    now = datetime.datetime.now()
    if format:
        return now.strftime(format)
    return now


def search_wikipedia(query):
    """Searches Wikipedia and returns the summary of the first result."""
    from wikipedia import summary

    try:
        # Limit to two sentences for brevity
        return summary(query, sentences=2)
    except:
        return "I couldn't find any information on that."


# Define Tools
tools = [
    StructuredTool(
        name="Time",
        func=get_current_time,
        description="Useful for when you need to know today's date.",
        args_schema=TimeInput  # ✅ Corrected args_schema
    ),
    StructuredTool(
        name="Wikipedia",
        func=search_wikipedia,
        description="Useful for when you need to know information about a topic.",
        args_schema=WikipediaInput  # ✅ Corrected args_schema
    ),
]

# Load the correct JSON Chat Prompt from the hub
prompt = hub.pull("hwchase17/structured-chat-agent")

# Initialize a ChatOpenAI model
llm = ChatOpenAI(model="gpt-4o-mini")

# Create a structured Chat Agent with Conversation Buffer Memory
# ConversationBufferMemory stores the conversation history, allowing the agent to maintain context across interactions
memory = ConversationBufferMemory(
    memory_key="chat_history", return_messages=True)

# create_structured_chat_agent initializes a chat agent designed to interact using a structured prompt and tools
# It combines the language model (llm), tools, and prompt to create an interactive agent
agent = create_structured_chat_agent(llm=llm, tools=tools, prompt=prompt)

# AgentExecutor is responsible for managing the interaction between the user input, the agent, and the tools
# It also handles memory to ensure context is maintained throughout the conversation
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    verbose=True,
    memory=memory,  # Use the conversation memory to maintain context
    handle_parsing_errors=True,  # Handle any parsing errors gracefully
    return_intermediate_steps=True  # Return the intermediate steps
)

# Initial system message to set the context for the chat
# SystemMessage is used to define a message from the system to the agent, setting initial instructions or context
initial_message = """You are an AI assistant that can provide helpful answers using available tools.
When you need to use a tool, always start your thought process by saying "I need to use the [tool name] tool because [reason]".
Then proceed with using the tool.

Available tools:
- Time: Use this when you need to know the current date and time
- Wikipedia: Use this when you need to find information about a topic

Always explain your reasoning before using any tool."""
memory.chat_memory.add_message(SystemMessage(content=initial_message))

# Chat Loop to interact with the user
while True:
    user_input = input("User: ")
    if user_input.lower() == "exit":
        break

    # Add the user's message to the conversation memory
    memory.chat_memory.add_message(HumanMessage(content=user_input))

    # Invoke the agent with the user input and the current chat history
    response = agent_executor.invoke({"input": user_input})
    
    # Display the intermediate steps (thought process)
    print("\nAgent Thought Process:")
    for i, (action, observation) in enumerate(response["intermediate_steps"], 1):
        print(f"\nStep {i}:")
        print(f"Thought: {action.log}")
        print(f"Action: {action.tool}")
        print(f"Input: {action.tool_input}")
        print(f"Observation: {observation}")
    
    print("\nFinal Response:")
    print("Bot:", response["output"])

    # Add the agent's response to the conversation memory
    memory.chat_memory.add_message(AIMessage(content=response["output"]))