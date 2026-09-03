# Importing Dependencies
from langchain.chat_models import init_chat_model, BaseChatModel
from langchain.agents.middleware import SummarizationMiddleware
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.messages import ToolMessage, HumanMessage
from langchain.agents import create_agent, AgentState
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.state import CompiledStateGraph
from langchain.tools import tool, ToolRuntime
from flight_agent import build_flight_agent
from hotel_agent import build_hotel_agent
from visa_agent import build_visa_agent
from langgraph.types import Command
from dotenv import load_dotenv
from os import environ

# Loading environment variables
load_dotenv()

# Defining the model name
model_name="gpt-5-nano"

# Instantiating the LLM
llm=init_chat_model(model=model_name)

# Defining agent state
class SupervisorState(AgentState):
    destination_country: str
    origin_country: str
    destination_city: str
    origin_city: str
    flight_class: str
    checked_baggage: int
    departure_date: str
    return_date: str
    preferred_airline: str | None = None
    n_passengers: int = 1
    hotel_filter: str = "highly rated"
    hotel_class: int = 5
    n_hotels: int = 5

# Defining an asynchronous function
async def build_supervisor_agent(llm: BaseChatModel = llm) -> CompiledStateGraph:
    # Instantiating the MCP client
    mcp_client=MultiServerMCPClient(connections={"tavily-mcp": {"command": "npx", 
                                                                "transport": "stdio",
                                                                "args": ["-y", "tavily-mcp"],
                                                                "env": {"TAVILY_API_KEY": environ.get("TAVILY_API_KEY")}},
                                                 "kiwi": {"transport": "streamable-http",
                                                          "url": "https://mcp.kiwi.com"}})

    # Extracting the MCP Tools
    mcp_tools=await mcp_client.get_tools()

    # Filtering the tools needed for the agent
    flight_tools=[i for i in mcp_tools if i.name=="search-flight"]
    
    # Filtering the tools needed for the agent
    web_search_tools=[i for i in mcp_tools if "tavily" in i.name]
    
    # Building the flight agent
    flight_agent=build_flight_agent(llm=llm, agent_tools=flight_tools)
    
    # Building the hotel agent
    hotel_agent=build_hotel_agent(llm=llm, agent_tools=web_search_tools)
    
    # Building the hotel agent
    visa_agent=build_visa_agent(llm=llm, agent_tools=web_search_tools)
    
    # Defining a tool to update the agent state
    @tool
    def update_state(destination_country: str,
                     origin_country: str,
                     destination_city: str,
                     origin_city: str,
                     flight_class: str,
                     checked_baggage: int,
                     departure_date: str,
                     return_date: str,
                     preferred_airline: str,
                     n_passengers: int,
                     hotel_filter: str,
                     hotel_class: int,
                     n_hotels: int,
                     runtime: ToolRuntime) -> Command:
        """
        Update the state when you know all of the values which are following:
        
        - destination_country
        - origin_country
        - destination_city
        - origin_city
        - flight_class
        - checked_baggage
        - departure_date
        - return_date
        - preferred_airline
        - n_passengers
        - hotel_filter
        - hotel_class
        - n_hotels
        """
        # Updating the state
        updated_state=Command(update={"destination_country": destination_country,
                                      "origin_country": origin_country,
                                      "destination_city": destination_city,
                                      "origin_city": origin_city,
                                      "flight_class": flight_class,
                                      "checked_baggage": checked_baggage,
                                      "departure_date": departure_date,
                                      "return_date": return_date,
                                      "preferred_airline": preferred_airline,
                                      "n_passengers": n_passengers,
                                      "hotel_filter": hotel_filter,
                                      "hotel_class": hotel_class,
                                      "n_hotels": n_hotels,
                                      "messages": [ToolMessage(content="Successfully updated the state", tool_call_id=runtime.tool_call_id)]})
        
        # Returning the updated state
        return updated_state
    
    # Defining a tool to invoke the visa agent
    @tool
    def consult_visa_agent(runtime: ToolRuntime) -> dict:
        """Consult the visa agent for destination requirements and restrictions."""
        # Extracting the destination country
        destination_country=runtime.get("destination_country").title()
        
        # Extracting the origin country
        origin_country=runtime.get("origin_country").title()
        
        # Constructing the user query
        user_query=f"As citizen of {origin_country}, what are the visa requirements to visit {destination_country}?"
        
        # Sending request to the agent
        response=visa_agent.invoke(input={"messages": [HumanMessage(content=user_query)]})
        
        # Extracting the structured output
        return response.get("structured_response")
    
    # Defining a tool to invoke the hotel agent
    @tool
    def consult_hotel_agent(runtime: ToolRuntime) -> str:
        """Consult the hotel agent to find accommodations."""
        # Extracting the destination city where a user wants to go to
        destination_city=runtime.get("destination_city").title()
        
        # Extracting the additional hotel filter
        hotel_filter=runtime.get("hotel_filter").lower()
        
        # Extracting the number of hotels
        n_hotels=runtime.get("n_hotels")
        
        # Extracting the hotel class
        hotel_class=runtime.get("hotel_class")
        
        # Constructing the user query
        user_query=f"Find the top {n_hotels} {hotel_filter} {hotel_class}-star hotels in {destination_city}."
                
        # Sending request to the agent
        response=hotel_agent.invoke(input={"messages": [HumanMessage(content=user_query)]})
        
        # Extracting the structured output
        return response.get("structured_response")
    
    # Defining a tool to invoke the flight agent
    @tool
    async def consult_flight_agent(runtime: ToolRuntime) -> dict:
        """Consult the flight agent to search for available flights."""
        # Extracting the destination city where a user wants to go to
        destination_city=runtime.get("destination_city").title()
        
        # Extracting the origin city where a user will depart from
        origin_city=runtime.get("origin_city").title()
        
        # Extracting the flight class (e.g., Economy & Business)
        flight_class=runtime.get("flight_class")
        
        # Extracting the number of checked baggage
        checked_baggage=runtime.get("checked_baggage")
        
        # Extracting the departure date
        departure_date=runtime.get("departure_date")
        
        # Extracting the return date
        return_date=runtime.get("return_date")
        
        # Extracting the preferred airline
        preferred_airline=runtime.get("preferred_airline")
        
        # Extracting the number of passengers
        n_passengers=runtime.get("preferred_airline")
        
        # Constructing the user query
        user_query=f"""
        Find the ticket based on the following criteria:
        
        From: {origin_city}
        To: {destination_city}
        Class: {flight_class}
        Departure Date: {departure_date}
        Return Date: {return_date}
        Number of Adults: {n_passengers}
        Number of Checked Baggage: {checked_baggage}
        Preferred Airline: {preferred_airline if preferred_airline else "No Preference"}
        """
        
        # Sending request to the agent
        response=await flight_agent.ainvoke(input={"messages": [HumanMessage(content=user_query)]})
        
        # Extracting the structured output
        return response.get("structured_response")
    
    # Instantiating the conversation summarization for the memory
    supervisor_memory_middleware=SummarizationMiddleware(model=llm,
                                                         trigger=("fraction", 0.75),
                                                         keep=("fraction", 0.25))
    
    # Defining a system prompt
    supervisor_system_prompt="""
    You are a travel planner.
    """
    
    # Building the supervisor agent
    supervisor_agent=create_agent(model=llm,
                                  tools=[update_state,
                                         consult_visa_agent,
                                         consult_hotel_agent,
                                         consult_flight_agent],
                                  system_prompt=supervisor_system_prompt,
                                  middleware=[supervisor_memory_middleware],
                                  state_schema=SupervisorState,
                                  checkpointer=InMemorySaver(),
                                  name="supervisor_agent")
    
    # Returning the agent
    return supervisor_agent