# Importing Dependencies
from langchain.agents.middleware import SummarizationMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.state import CompiledStateGraph
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from pydantic import Field, BaseModel
from typing import List, Optional
from dotenv import load_dotenv

# Loading the environment variables
load_dotenv()

# Defining the model name
model_name="gpt-5-nano"

# Instantiating the LLM
llm=init_chat_model(model=model_name)

# Defining an asynchronous function
def build_hotel_agent(agent_tools: list) -> CompiledStateGraph:
    # Defining a system prompt
    hotel_system_prompt=f"""
    You are a hotel search expert. You are extremely capable of
    finding detailed information based on a user criteria.
    You MUST use '{agent_tools[0].name}' tool to answer user's questions regarding hotels.
    When using the '{agent_tools[0].name}' tool, prioritize querying domains like 'https://www.tripadvisor.com/', 
    'https://www.booking.com/', and 'https://www.agoda.com/' to extract current user ratings, exact review counts, and facility details.
    Your answer must be precise regarding the requirements and your tone must be professional. You MUST strictly output the results conforming to the requested schema.

    CRITICAL SEARCH RULE 1: You must cross-reference data. You are FORBIDDEN from relying on a single platform. 
    You MUST pull data from at least TWO of the following domains: 'https://www.tripadvisor.com/', 'https://www.booking.com/', or 'https://www.agoda.com/' for your results.

    CRITICAL SEARCH RULE 2: Not all aggregator websites list official hotel URLs. After identifying the target hotels, you MUST perform a separate, dedicated search for each 
    hotel's name (e.g., "JW Marriott Absheron Baku official website") to find its official 'website' link.

    CRITICAL RULE: Populate the final output with the actual scraped hotel data. Never output JSON schema definitions, field types, or descriptions inside the values.
    """

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

    # Instantiating the conversation summarization for the memory
    hotel_memory_middleware=SummarizationMiddleware(model=llm,
                                                    trigger=("fraction", 0.75),
                                                    keep=("fraction", 0.25))

    # Instantiating the agent
    hotel_agent=create_agent(model=llm,
                             tools=agent_tools,
                             system_prompt=hotel_system_prompt,
                             middleware=[hotel_memory_middleware],
                             response_format=HotelAgentResponse,
                             checkpointer=InMemorySaver(),
                             name="hotel_agent")
    
    # Returning the agent
    return hotel_agent