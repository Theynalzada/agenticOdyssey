# Importing Dependencies
from langchain_core.language_models import BaseChatModel
from langchain_mcp_adapters.tools import StructuredTool
from langgraph.graph.state import CompiledStateGraph
from langchain.messages import SystemMessage
from langchain.agents import create_agent
from pydantic import Field, BaseModel
from typing import List, Optional

# Instantiating the response schema for flight that meets the conditions
class FlightAgentResponse(BaseModel):
    booking_links: List[str] = Field(default_factory=list, description="The list of booking URLs found, based on the specifications from the user, ordered from best match first (lowest price, if determinable from the search results). Empty if no matching flights were found — do not fabricate a placeholder entry in this case.")
    message: Optional[str] = Field(default=None, description="A short note for the user, populated only when booking_links is empty — e.g. explaining that no flights matched their criteria. Leave empty when booking_links is populated.")
 
# Defining an asynchronous function
async def build_flight_agent(llm: BaseChatModel, agent_tools: List[StructuredTool]) -> CompiledStateGraph:
    # Defining a system prompt
    flight_system_prompt="""
    You are a precise flight search expert. You are extremely capable of finding tickets based
    on the specifications provided by a user.
    Your answer must be precise regarding the requirements and your tone must be professional.
 
    CRITICAL RULES:
        1. Base your response STRICTLY on the real data returned by your search tools and NEVER
           invent, guess, or hallucinate booking links.
        2. TOOL PARAMETER MAPPING — push constraints into the search tool's own filters rather
           than filtering results yourself after the fact:
             - currency: always pass 'USD' explicitly.
             - Preferred/required airline(s): convert the airline name(s) to their correct
               2-letter IATA code(s) and pass them via select_airlines (comma-separated, e.g.
               'SQ'). If you cannot confidently determine the IATA code for a named airline, do
               not guess — proceed without select_airlines and instead verify the airline name
               against the results yourself (see rule 5).
             - Cabin class: map to the tool's single-letter codes — Economy=M, Premium Economy=W,
               Business=C, First=F.
             - Dates: convert to dd/mm/yyyy format with forward slashes (e.g. 20/09/2026), taking
               care not to transpose day and month.
             - Checked/cabin baggage: these are arrays with ONE ENTRY PER PASSENGER, not a single
               total count. E.g. for 2 adults where only one has a checked bag, pass
               adults_hold_bags=[1, 0], not a scalar.
             - Sorting: pass sort='price' so the tool itself returns cheapest-first, rather than
               re-ranking results yourself.
        3. If the same booking link appears more than once across your search results, include
           it only once in booking_links.
        4. Order booking_links in the order returned by the tool (best match / lowest price
           first, per rule 2's sort parameter).
        5. STRICT CONSTRAINT MATCHING (safety net): even after applying tool-level filters, only
           include flight options that actually satisfy every explicitly stated user requirement
           — airline, cabin class, stops, baggage, or otherwise. Do not substitute a flight that
           violates a stated constraint just because it exists for the same route. If zero
           results satisfy every explicitly stated constraint, this counts as "no flights found"
           even if other, non-matching flights are available for the route.
        6. If you fail to find tickets matching the user's criteria, leave booking_links empty
           and set message to 'Sorry, there is no flight based on your specifications.' Do not
           put this message inside booking_links.
        7. Strictly output the results conforming to the requested schema.
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