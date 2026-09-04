# Importing Dependencies
from langchain_core.language_models import BaseChatModel
from langchain_mcp_adapters.tools import StructuredTool
from langgraph.graph.state import CompiledStateGraph
from langchain.messages import SystemMessage
from langchain.agents import create_agent
from pydantic import Field, BaseModel
from typing import List, Optional

# Defining the details for a particular hotel
class HotelDetails(BaseModel):
    name: str = Field(description="The name of the hotel (e.g., Courtyard by Marriott Baku)")
    address: str = Field(description="The full address of the hotel")
    city: str = Field(description="The city where the hotel is located")
    country: str = Field(description="The country where the hotel is located. ALWAYS return the FULL country name (e.g., Azerbaijan)")
    stars: Optional[int] = Field(default=None, description="The star classification of the hotel (e.g., 5), if the property has an official star rating. Omit for boutique/unrated properties rather than guessing.")

# Defining the response format for the agent
class HotelAgentResponse(BaseModel):
    hotels: List[HotelDetails] = Field(description="The list of matching hotels extracted from the search results.")

# Defining an asynchronous function
async def build_hotel_agent(llm: BaseChatModel, agent_tools: List[StructuredTool]) -> CompiledStateGraph:
    # Defining a system prompt
    hotel_system_prompt="""
    You are a hotel search expert. You are extremely capable of
    finding detailed information based on a user criteria.
    Your answer must be precise regarding the requirements and
    your tone must be professional.
    You MUST strictly output the results conforming to the requested schema.

    - If the same hotel appears more than once across your searches, merge it into a single
      entry rather than listing it twice.

    CRITICAL RULE: Populate the final output with the actual scraped hotel data.
    Never output JSON schema definitions, field types, or descriptions inside the values.
    """

    # Creating a system message
    hotel_system_message=SystemMessage(content=hotel_system_prompt)

    # Instantiating the agent
    hotel_agent=create_agent(model=llm,
                             tools=agent_tools,
                             system_prompt=hotel_system_message,
                             response_format=HotelAgentResponse,
                             name="hotel_agent")

    # Returning the agent
    return hotel_agent