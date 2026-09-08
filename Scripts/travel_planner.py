# Importing Dependencies
from langchain.agents.middleware import dynamic_prompt, ModelRequest, SummarizationMiddleware
from langchain.messages import ToolMessage, HumanMessage, SystemMessage
from Scripts.flight_agent import build_flight_agent, FlightAgentResponse
from Scripts.hotel_agent import build_hotel_agent, HotelAgentResponse
from Scripts.visa_agent import build_visa_agent, VisaAgentResponse
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent, AgentState
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.state import CompiledStateGraph
from langchain.chat_models import init_chat_model
from langchain.tools import tool, ToolRuntime
from pydantic import Field, BaseModel
from langgraph.types import Command
from dataclasses import dataclass
from dotenv import load_dotenv
from typing import Optional
from os import environ

# Loading environment variables
load_dotenv()

# Defining the model name
model_name="gpt-5-nano"

# Instantiating the LLM
llm=init_chat_model(model=model_name)

# Creating a context for the agent
@dataclass
class SupervisorContext:
  user_language: str = "English"

# Creating a dynamic prompt
@dynamic_prompt
def update_system_prompt(request: ModelRequest) -> SystemMessage:
  """Update the language that will be used throughout conversation based on user's request."""
  # Extracting the user language
  user_language=request.runtime.context.user_language
  
  # Extracting the user language
  system_prompt=request.system_message.content
  
  # Creating a condition based on the user language
  if user_language!="English":
    # Updating the system prompt in case the user language is not English
    system_prompt+=f"You MUST communicate with the user in {user_language} throughout the conversation."
    
  # Returning the system prompt
  return SystemMessage(content=system_prompt)
  
# Defining the agent state
class SupervisorState(AgentState):
    destination_country: Optional[str]
    countryFrom: Optional[str]
    destination_city: Optional[str]
    cityFrom: Optional[str]
    departure_date: Optional[str]
    return_date: Optional[str]
    preferred_airline: Optional[str]
    n_hotels: Optional[int]
    stars: Optional[int]
    cabin_class: Optional[str]
    n_adults: Optional[int]
    n_children: Optional[int]
    n_infants: Optional[int]
    n_checked_baggage: Optional[int]

# Defining the response format for the supervisor agent
class SupervisorAgentResponse(BaseModel):
    message_to_user: Optional[str] = Field(default=None, description="Any conversational text for the user — a clarifying question when required trip details are still missing, or a brief wrap-up note once results are populated below. Do NOT restate or paraphrase the structured data in visa_info/hotel_info/flight_info here; that data speaks for itself.")
    visa_info: Optional[VisaAgentResponse] = Field(default=None, description="The visa agent's structured response, populated exactly as returned once consult_visa_agent has been called. Leave as None until then.")
    hotel_info: Optional[HotelAgentResponse] = Field(default=None, description="The hotel agent's structured response, populated exactly as returned once consult_hotel_agent has been called. Leave as None until then.")
    flight_info: Optional[FlightAgentResponse] = Field(default=None, description="The flight agent's structured response, populated exactly as returned once consult_flight_agent has been called. Leave as None until then.")
    
# Defining an asynchronous function
async def build_supervisor_agent() -> CompiledStateGraph:
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
    flight_agent=await build_flight_agent(llm=llm, agent_tools=flight_tools)
    
    # Building the hotel agent
    hotel_agent=await build_hotel_agent(llm=llm, agent_tools=web_search_tools)
    
    # Building the hotel agent
    visa_agent=await build_visa_agent(llm=llm, agent_tools=web_search_tools)
    
    # Defining a tool to update the state
    @tool
    def update_state(destination_country: Optional[str] = None,
                     countryFrom: Optional[str] = None,
                     destination_city: Optional[str] = None,
                     cityFrom: Optional[str] = None,
                     n_hotels: Optional[int] = None,
                     stars: Optional[int] = None,
                     cabin_class: Optional[str] = None,
                     n_adults: Optional[int] = None,
                     n_children: Optional[int] = None,
                     n_infants: Optional[int] = None,
                     n_checked_baggage: Optional[int] = None,
                     departure_date: Optional[str] = None,
                     return_date: Optional[str] = None,
                     preferred_airline: Optional[str] = None,
                     runtime: ToolRuntime = None) -> Command:
        """
        Record trip details as they are revealed during the conversation. Only pass the
        fields the user has actually specified in THIS turn — leave every other field as
        None. Previously stored values are preserved for any field you omit; they are only
        overwritten when you explicitly pass a new value.
 
        Fields: destination_country, destination_city, cityFrom, countryFrom (the user's
        own nationality/citizenship, needed for visa lookups), n_hotels, stars, cabin_class,
        n_adults, n_children, n_infants, n_checked_baggage, departure_date, return_date,
        preferred_airline.
        """
        # Creating a state dictionary
        state_dict={"destination_country": destination_country,
                    "countryFrom": countryFrom,
                    "destination_city": destination_city,
                    "cityFrom": cityFrom,
                    "n_hotels": n_hotels,
                    "stars": stars,
                    "cabin_class": cabin_class,
                    "n_adults": n_adults,
                    "n_children": n_children,
                    "n_infants": n_infants,
                    "n_checked_baggage": n_checked_baggage,
                    "departure_date": departure_date,
                    "return_date": return_date,
                    "preferred_airline": preferred_airline}
        
        # Updating the state based on passed values
        updated_state_dict={k: v for k, v in state_dict.items() if v is not None}
        
        # Adding the tool message
        updated_state_dict["messages"]=[ToolMessage(content="Updated state successfully!", tool_call_id=runtime.tool_call_id)]
        
        # Returning the updated state
        return Command(update=updated_state_dict)
    
    # Defining a tool to invoke the visa agent
    @tool
    async def consult_visa_agent(runtime: ToolRuntime) -> dict:
        """Consult the visa agent for destination requirements and restrictions."""
        # Extracting the destination country
        destination_country=runtime.state.get("destination_country")
        
        # Extracting the origin country
        countryFrom=runtime.state.get("countryFrom")
        
        # Guarding against missing prerequisites instead of crashing
        if not destination_country or not countryFrom:
            return {"error": "Missing destination_country or countryFrom. Call update_state with these fields first."}
          
        # Extracting the user language from the context
        user_language=runtime.context.user_language
        
        # Constructing the user query
        user_query=f"""
        As citizen of {countryFrom.title()}, what are the visa requirements to visit {destination_country.title()}?
        
        IMPORTANT:
        Provide all textual information in {user_language}.
        """
        
        # Sending request to the agent
        response=await visa_agent.ainvoke(input={"messages": [HumanMessage(content=user_query)]})
        
        # Extracting the structured response
        structured_response=response.get("structured_response")
        
        # Extracting the structured output
        return structured_response.model_dump() if structured_response else {"error": "Visa agent did not return a structured response."}
    
    # Defining a tool to invoke the hotel agent
    @tool
    async def consult_hotel_agent(runtime: ToolRuntime) -> dict:
        """Consult the hotel agent to find accommodations."""
        # Extracting the destination city where a user wants to go to
        destination_city=runtime.state.get("destination_city")
        
        # Extracting the number of hotels
        n_hotels=runtime.state.get("n_hotels")
        
        # Extracting the hotel star
        stars=runtime.state.get("stars")
        
        # Guarding against missing prerequisites instead of crashing
        if not destination_city:
            return {"error": "Missing destination_city. Call update_state with this field first."}
        
        # Guardrail against the number of hotels
        n_hotels=n_hotels if n_hotels else 5
        
        # Guardrail against the number of stars
        stars=stars if stars else 5
        
        # Constructing the user query
        user_query=f"""
        Find {n_hotels} {stars}-star hotels in {destination_city.title()}.
        """
                
        # Sending request to the agent
        response=await hotel_agent.ainvoke(input={"messages": [HumanMessage(content=user_query)]})
        
        # Extracting the structured response
        structured_response=response.get("structured_response")
        
        # Extracting the structured output
        return structured_response.model_dump() if structured_response else {"error": "Hotel agent did not return a structured response."}
    
    # Defining a tool to invoke the flight agent
    @tool
    async def consult_flight_agent(runtime: ToolRuntime) -> dict:
        """Consult the flight agent to search for available flights."""
        # Extracting the destination city where a user wants to go to
        cityTo=runtime.state.get("destination_city")
        
        # Extracting the city where a user will depart from
        cityFrom=runtime.state.get("cityFrom")
        
        # Extracting the departure date
        departure_date=runtime.state.get("departure_date")
        
        # Guarding against missing prerequisites instead of crashing
        if not cityTo or not cityFrom or not departure_date:
            return {"error": "Missing destination_city, cityFrom, or departure_date. Call update_state with these fields first."}
        
        # Extracting the return date
        return_date=runtime.state.get("return_date")
        
        # Extracting the cabin class
        cabin_class=runtime.state.get("cabin_class") or "Economy"
        
        # Extracting the number of checked baggage
        n_checked_baggage=runtime.state.get("n_checked_baggage") or 0
        
        # Extracting the preferred airline
        preferred_airline=runtime.state.get("preferred_airline")
        
        # Extracting the number of adults
        n_adults=runtime.state.get("n_adults") or 1
        
        # Extracting the number of children
        n_children=runtime.state.get("n_children") or 0
        
        # Extracting the number of infants
        n_infants=runtime.state.get("n_infants") or 0
        
        # Constructing the user query
        user_query=f"""
        Find tickets based on the following criteria:
        
        From: {cityFrom.title()}
        To: {cityTo.title()}
        Cabin Class: {cabin_class}
        Departure Date: {departure_date}
        Return Date: {return_date}
        Number of Adults: {n_adults}
        Number of Children: {n_children}
        Number of Infants: {n_infants}
        Number of Checked Baggage: {n_checked_baggage}
        Preferred Airline: {preferred_airline if preferred_airline else "Any"}
        """
        
        # Sending request to the agent
        response=await flight_agent.ainvoke(input={"messages": [HumanMessage(content=user_query)]})
        
        # Extracting the structured response
        structured_response=response.get("structured_response")
        
        # Extracting the structured output
        return structured_response.model_dump() if structured_response else {"error": "Flight agent did not return a structured response."}
    
    # Instantiating the conversation summarization for the memory
    supervisor_memory_middleware=SummarizationMiddleware(model=llm,
                                                         trigger=("fraction", 0.75),
                                                         keep=("fraction", 0.25))
    
    # Defining a system prompt
    supervisor_system_prompt="""
    You are a travel planner who helps a user plan a trip involving visa requirements, hotels,
    and flights.
 
    WORKFLOW:
    - Collect trip details from the conversation incrementally. Every time the user reveals a
      new detail (destination, dates, passenger counts, preferences, etc.), call update_state
      with ONLY the fields that were just revealed — never guess or invent values for fields
      the user hasn't mentioned.
    - PROACTIVE CONSULTATION: the moment a consult_* tool's prerequisites are satisfied, you
      MUST call it — do not wait for the user to explicitly ask for visa information, hotel
      options, or flight options separately. If the user only mentions a destination and their
      nationality, for example, that's enough to trigger consult_visa_agent immediately, even
      if they didn't use the word "visa." Treat "plan my trip" as an implicit request for all
      three, not just whichever one the user's phrasing happened to emphasize.
    - Do not re-call a consult_* tool you've already successfully called for the current trip
      unless the relevant details have since changed (e.g. a different destination or dates) —
      avoid redundant re-consultation of unchanged information.
    - Before calling consult_visa_agent, make sure destination_country and countryFrom are
      both known. If either is missing, use message_to_user to ask for it, leave visa_info as
      None, and do not call the tool yet.
    - Before calling consult_hotel_agent, make sure destination_city is known. n_hotels and
      stars are optional preferences — proceed without them if the user hasn't specified.
    - Before calling consult_flight_agent, make sure cityFrom, destination_city, and
      departure_date are all known. Use message_to_user to ask for any that are missing rather
      than guessing, and leave flight_info as None until they're available.
    - If a consult_* tool returns an "error" key, that means required information is still
      missing — ask the user for it via message_to_user, then retry the tool once you have it.
 
    OUTPUT FORMAT:
    - You MUST always respond with the structured TravelPlannerResponse schema, on every turn —
      including turns where you are only asking a clarifying question.
    - visa_info, hotel_info, and flight_info must be populated with EXACTLY the structured data
      returned by the corresponding consult_* tool, unmodified — do not paraphrase, summarize,
      reformat, or drop any fields from what the tool returned. Leave a field as None if that
      agent has not been consulted yet in this conversation.
    - message_to_user carries any conversational text: a clarifying question when information is
      missing, or a short, non-redundant wrap-up note once results are populated. Never restate
      the contents of visa_info/hotel_info/flight_info in message_to_user — that data is already
      structured and visible; message_to_user is only for things the structured fields can't say.
 
    Your tone must be professional, helpful, and clear.
    """
    # Creating a system message
    supervisor_system_message=SystemMessage(content=supervisor_system_prompt)
    
    # Building the supervisor agent
    supervisor_agent=create_agent(model=llm,
                                  tools=[update_state,
                                         consult_visa_agent,
                                         consult_hotel_agent,
                                         consult_flight_agent],
                                  system_prompt=supervisor_system_message,
                                  middleware=[update_system_prompt, 
                                              supervisor_memory_middleware],
                                  response_format=SupervisorAgentResponse,
                                  state_schema=SupervisorState,
                                  context_schema=SupervisorContext,
                                  checkpointer=InMemorySaver(),
                                  name="supervisor_agent")
    
    # Returning the agent
    return supervisor_agent