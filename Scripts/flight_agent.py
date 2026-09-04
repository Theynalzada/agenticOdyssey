# Importing Dependencies
from langchain_core.language_models import BaseChatModel
from langchain_mcp_adapters.tools import StructuredTool
from langgraph.graph.state import CompiledStateGraph
from langchain.messages import SystemMessage
from typing import List, Literal, Optional
from langchain.agents import create_agent
from pydantic import BaseModel, Field

# Instantiating the response schema based on the flight segment
class FlightSegment(BaseModel):
    departure_airport: str = Field(description="The full name of the departure airport (e.g., Heydar Aliyev International)")
    departure_airport_code: Optional[str] = Field(default=None, description="The 3-letter IATA code (e.g., GYD)")
    departure_time: str
    duration: str = Field(description="The duration of this specific segment (e.g., '3h 5m')")
    carrier: str = Field(description="The full airline name (e.g., 'Azerbaijan Airlines'). You must convert and write the full name, NEVER output 2-letter IATA codes like 'J2'.")
    operating_carrier: Optional[str] = Field(default=None, description="Only if operated on behalf of another carrier")
    arrival_airport: str = Field(description="The full name of the arrival airport (e.g., Istanbul Airport)")
    arrival_airport_code: Optional[str] = Field(default=None, description="The 3-letter IATA code (e.g., IST, SAW)")
    arrival_time: str

# Instantiating the resonse schema based on the journey
class Journey(BaseModel):
    journey_type: str = Field(description="e.g., 'Outbound' or 'Inbound'")
    route: str = Field(description="The overall city-to-city route (e.g., 'Baku --> London'). You MUST use city names, NEVER full airport names.")
    total_duration: str = Field(description="The total duration of the journey, including layovers")
    segments: List[FlightSegment] = Field(description="The individual physical flights that make up this journey")

# Instantiating the response schema for flight that meets the conditions
class FlightOption(BaseModel):
    total_price: float = Field(description="The price of a ticket")
    currency: Optional[Literal["EUR", "USD"]] = Field(default="USD", description="The currency the ticket price calculated with. If the currency has not been specified then you MUST default to 'USD'.")
    n_adults: Optional[int] = Field(default=1, description="The number of adults for this flight specified by the user. If the number of adults has not been specified then you MUST default to 1.")
    n_children: Optional[int] = Field(default=0, description="The number of children aged between 2-11 for this flight specified by the user. If the number of children has not been specified then you MUST default to 0.")
    n_infants:  Optional[int] = Field(default=0, description="The number of infants aged between 0-2 this flight specified by the user. If the number of infants has not been specified then you MUST default to 0.")
    n_checked_baggage: Optional[int] = Field(default=0, description="The number of checked baggage by the user. If the number of checked baggage has not been specified then you MUST default to 0.")
    n_cabin_baggage: Optional[int] = Field(default=0, description="The number of cabin baggage by the user. If the number of cabin baggage has not been specified then you MUST default to 0.")
    cabin_class: Optional[Literal["Economy", "Premium Economy", "Business", "First"]] = Field(default="Economy", description="The cabin class the user wants to fly with. If the cabin class has not been specified, then you MUST default to 'Economy'.")
    booking_link: str = Field(description="URL for booking")
    journeys: List[Journey] = Field(description="The directional journeys (e.g., Outbound and Inbound) that make up this flight option.")

# Instantiating the response schema for all flights that meet the conditions
class FlightAgentResponse(BaseModel):
    flight_options: List[FlightOption]

# Defining an asynchronous function
async def build_flight_agent(llm: BaseChatModel, agent_tools: List[StructuredTool]) -> CompiledStateGraph:
    # Defining a system prompt
    flight_system_prompt="""
    You are a precise flight search expert.

    CRITICAL RULES:
        1. Base your response STRICTLY on the real data returned by your search tools.
        2. NEVER invent, guess, or hallucinate times, prices, or airlines.
        3. PREVENT DATA DUPLICATION: When parsing multiple flight options, process each one individually. Pay strict attention to exact prices, departure/arrival times, and specific airports (e.g., IST vs SAW). DO NOT blindly copy-paste data from one flight option to another.
        4. You MUST parse both OUTBOUND and INBOUND flights with the correct data 
        5. HIERARCHICAL PARSING: A 'FlightOption' consists of 'Journeys' (Outbound/Inbound). Each 'Journey' consists of one or more 'FlightSegments' (the actual physical flights). Accurately map layovers into multiple segments within a journey.
        6. You MUST always use the default value if user has not provided any for a particular search criteria.
        7. When executing the flight search tool, you MUST explicitly pass 'USD' as the currency parameter.
        8. Return the top 3 most relevant flight options that satisfy the criteria.
        9. Strictly output the results conforming to the requested schema.

    Your answer must be precise regarding the requirements and your tone must be professional.
    """
    
    # Creating a system message
    flight_system_message=SystemMessage(content=flight_system_prompt)

    # Instantiating the agent
    flight_agent=create_agent(model=llm,
                              tools=agent_tools,
                              system_prompt=flight_system_message,
                              response_format=FlightAgentResponse,
                              name="flight_agent")
    
    # Returning the flight agent
    return flight_agent