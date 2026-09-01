# Importing Dependencies
from langchain.agents.middleware import SummarizationMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.state import CompiledStateGraph
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from pydantic import Field, BaseModel
from typing import Optional, List
from dotenv import load_dotenv

# Loading environment variables
load_dotenv()

# Defining the model name
model_name="gpt-5-nano"

# Instantiating the LLM
llm=init_chat_model(model=model_name)

# Defining an asynchronous function
def build_visa_agent(agent_tools: list) -> CompiledStateGraph:
  # Defining a response format
  class VisaAgentResponse(BaseModel):
    user_nationality: str = Field(description="The nationality of a user")
    destination_country: str = Field(description="The country a user wants to visit")
    requires_visa: bool = Field(description="Whether the country a user wants to visit requires visa or not")
    requirements: Optional[List[str]] = Field(default=None, description="The list of requirements to apply for a Schengen visa if a user is required to have it")
    schengen_countries: Optional[List[str]] = Field(default=None, description="The list of Schengen countries whose visa is accepted in the country a user wants to visit")
    source: Optional[List[str]] = Field(default=None, description="The list of sources a user can use to get more information regarding visa acquiring process")
  
  # Defining a system prompt
  visa_system_prompt=f"""
  You are a visa expert. You are extremely capable of
  finding detailed information based on visa requirements of a country a user wants to visit. 
  Focus your search and response on this specific country.
  You MUST use '{agent_tools[0].name}' tool to answer user's questions regarding visa.
  Your answer must be precise regarding the requirements and your tone must be professional. 
  You MUST strictly output the results conforming to the requested schema.

  CRITICAL RULE: Populate the final output with the actual scraped visa data. 
  Never output JSON schema definitions, field types, or descriptions inside the values.
  """

  # Instantiating the conversation summarization for the memory
  visa_memory_middleware=SummarizationMiddleware(model=llm,
                                                 trigger=("fraction", 0.75),
                                                 keep=("fraction", 0.25))

  # Defining an agent that checks visa availability
  visa_agent=create_agent(model=llm,
                          tools=agent_tools,
                          system_prompt=visa_system_prompt,
                          middleware=[visa_memory_middleware],
                          response_format=VisaAgentResponse,
                          checkpointer=InMemorySaver(),
                          name="visa_agent")
  
  # Returning the agent
  return visa_agent