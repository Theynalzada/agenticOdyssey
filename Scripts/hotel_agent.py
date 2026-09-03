# Importing Dependencies
from langchain_mcp_adapters.tools import StructuredTool
from langgraph.graph.state import CompiledStateGraph
from langchain.chat_models import BaseChatModel
from langchain.agents import create_agent
from pydantic import Field, BaseModel
from typing import List, Optional

# Defining the details for a particular hotel
class HotelDetails(BaseModel):
    name: str = Field(description="The name of the hotel (e.g., JW Marriott Absheron Baku)")
    address: str = Field(description="The full address of the hotel")
    city: str = Field(description="The city where the hotel is located")
    country: str = Field(description="The country where the hotel is located")
    rating: float = Field(description="The user rating of the hotel (e.g., 4.8)")
    hotel_class: str = Field(description="Star classification (e.g., 5-star)")
    reviews: Optional[int] = Field(default=None, description="The approximate review count accross all platforms it was found on. Make sure to return only the aggregated number of reviews.")
    closest_landmarks: Optional[List[str]] = Field(default=None, description="Nearby landmarks")
    facilities: Optional[List[str]] = Field(default=None, description="Key amenities and facilities")
    website: Optional[str] = Field(default=None, description="Official website URL of the hotel")
    source: Optional[List[str]] = Field(default=None, description="Source URLs referenced for the data. Use ONLY the EXACT URLs and NEVER include an offical website URL to the source.")

# Defining the response format for the agent
class HotelAgentResponse(BaseModel):
    hotels: List[HotelDetails] = Field(description="The list of matching hotels extracted from the search results.")

# Defining an asynchronous function
def build_hotel_agent(llm: BaseChatModel, agent_tools: List[StructuredTool]) -> CompiledStateGraph:
    # Defining a system prompt
    hotel_system_prompt=f"""
    You are a hotel search expert. You are extremely capable of
    finding detailed information based on a user criteria.
    When searching for hotels, prioritize querying domains like 'https://www.tripadvisor.com/', 
    'https://www.booking.com/', and 'https://www.agoda.com/' to extract current user ratings, exact review counts, and facility details.
    Your answer must be precise regarding the requirements and your tone must be professional. You MUST strictly output the results conforming to the requested schema.

    CRITICAL SEARCH RULE 1: You must cross-reference data. You are FORBIDDEN from relying on a single platform. 
    You MUST pull data from at least TWO of the following domains: 'https://www.tripadvisor.com/', 'https://www.booking.com/', or 'https://www.agoda.com/' for your results.

    CRITICAL SEARCH RULE 2: Not all aggregator websites list official hotel URLs. After identifying the target hotels, you MUST perform a separate, dedicated search for each 
    hotel's name (e.g., "JW Marriott Absheron Baku official website") to find its official 'website' link.

    CRITICAL RULE: Populate the final output with the actual scraped hotel data. Never output JSON schema definitions, field types, or descriptions inside the values.
    """

    # Instantiating the agent
    hotel_agent=create_agent(model=llm,
                             tools=agent_tools,
                             system_prompt=hotel_system_prompt,
                             response_format=HotelAgentResponse,
                             name="hotel_agent")
    
    # Returning the agent
    return hotel_agent