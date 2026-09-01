# Importing Dependencies
from langchain.agents.middleware import SummarizationMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.state import CompiledStateGraph
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from pydantic import BaseModel, Field
from typing import List, Optional
from dotenv import load_dotenv

# Loading environment variables
load_dotenv()

# Defining the model name
model_name="gpt-5-nano"

# Instantiating the LLM
llm=init_chat_model(model=model_name)

# Instantiating the response schema for tickets
class TicketDetails(BaseModel):
    ticket_type: str = Field(description="e.g., Departure or Return")
    route: str = Field(description="The city-to-city route (e.g., 'Baku --> London'). You MUST use city names, NEVER full airport names.")
    departure_airport: str = Field(description="The full name of the departure airport (e.g., Heydar Aliyev International)")
    departure_airport_code: str = Field(description="The 3-letter IATA code (e.g., GYD)")
    departure_time: str
    duration: str
    carrier: str = Field(description="The full airline name (e.g., 'Azerbaijan Airlines'). You must convert and write the full name, NEVER output 2-letter IATA codes like 'J2'.")
    operating_carrier: Optional[str] = Field(default=None, description="Only if operated on behalf of another carrier")
    connection_number: Optional[str] = Field(default=None, description="The flight number or connection ID (e.g., J2 8105)")
    operating_connection_number: Optional[str] = Field(default=None, description="Only if operated on behalf of another carrier")
    arrival_airport: str = Field(description="The full name of the arrival airport (e.g., Gatwick Airport)")
    arrival_airport_code: str = Field(description="The 3-letter IATA code (e.g., LGW, LHR)")
    arrival_time: str

# Instantiating the response schema for flights
class FlightAgentResponse(BaseModel):
    total_price: str
    number_of_passengers: str
    flight_class: str = Field(description="e.g., Economy or Business")
    booking_link: str = Field(description="URL for booking")
    tickets: List[TicketDetails]

# Defining an asynchronous function
async def build_flight_agent(agent_tools: list) -> CompiledStateGraph:
    # Defining a system prompt
    flight_system_prompt=f"""
    You are a precise flight search expert.
    Search for flights using '{agent_tools[0].name}' tool. 
    Your answer must be precise regarding the requirements and your tone must be professional. 
    You MUST strictly output the results conforming to the requested schema.
    """

    # Instantiating the conversation summarization for the memory
    flight_memory_middleware=SummarizationMiddleware(model=llm,
                                                     trigger=("fraction", 0.75),
                                                     keep=("fraction", 0.25))

    # Instantiating the agent
    flight_agent=create_agent(model=llm,
                              tools=agent_tools,
                              system_prompt=flight_system_prompt,
                              middleware=[flight_memory_middleware],
                              response_format=FlightAgentResponse,
                              checkpointer=InMemorySaver(),
                              name="flight_agent")
    
    # Returning the flight agent
    return flight_agent