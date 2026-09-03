# Importing Dependencies
from langchain_mcp_adapters.tools import StructuredTool
from langgraph.graph.state import CompiledStateGraph
from langchain.chat_models import BaseChatModel
from langchain.agents import create_agent
from pydantic import Field, BaseModel
from typing import Optional, List

# Defining a response format
class VisaAgentResponse(BaseModel):
  user_nationality: str = Field(description="The nationality of a user")
  destination_country: str = Field(description="The country a user wants to visit")
  requires_visa: bool = Field(description="Whether the country a user wants to visit requires visa or not")
  requirements: Optional[List[str]] = Field(default=None, description="The list of requirements to apply for a Schengen visa if a user is required to have it")
  schengen_countries: Optional[List[str]] = Field(default=None, description="The list of Schengen countries whose visa is accepted in the country a user wants to visit")
  source: Optional[List[str]] = Field(default=None, description="The list of sources a user can use to get more information regarding visa acquiring process")

# Defining an asynchronous function
def build_visa_agent(llm: BaseChatModel, agent_tools: List[StructuredTool]) -> CompiledStateGraph:
  # Defining a system prompt
  visa_system_prompt=f"""
  You are a visa expert. You are extremely capable of
  finding detailed information based on visa requirements of a country a user wants to visit. 
  Focus your search and response on this specific country.
  Your answer must be precise regarding the requirements and your tone must be professional. 
  You MUST strictly output the results conforming to the requested schema.

  CRITICAL RULE: Populate the final output with the actual scraped visa data. 
  Never output JSON schema definitions, field types, or descriptions inside the values.
  """

  # Defining an agent that checks visa availability
  visa_agent=create_agent(model=llm,
                          tools=agent_tools,
                          system_prompt=visa_system_prompt,
                          response_format=VisaAgentResponse,
                          name="visa_agent")
  
  # Returning the agent
  return visa_agent