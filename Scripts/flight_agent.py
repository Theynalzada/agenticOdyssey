# Importing Dependencies
from langchain_core.language_models import BaseChatModel
from langchain_mcp_adapters.tools import StructuredTool
from pydantic import model_validator, Field, BaseModel
from langgraph.graph.state import CompiledStateGraph
from langchain.messages import SystemMessage
from typing import List, Literal, Optional
from langchain.agents import create_agent

# Instantiating the response schema based on the flight segment
class FlightSegment(BaseModel):
    departure_airport: str = Field(description="The full name of the departure airport (e.g., Heydar Aliyev International). Copy this verbatim from the tool output - never infer or reuse a value from a different segment.")
    departure_airport_code: Optional[str] = Field(default=None, description="The 3-letter IATA code (e.g., GYD)")
    departure_time: str = Field(description="Exact departure time as returned by the tool, in the same format the tool provided it.")
    duration: str = Field(description="The duration of this specific segment (e.g., '3h 5m'), taken from the tool output - do not compute or estimate it yourself.")
    carrier: str = Field(description="The full airline name (e.g., 'Azerbaijan Airlines'). You must convert and write the full name, NEVER output 2-letter IATA codes like 'J2'.")
    arrival_airport: str = Field(description="The full name of the arrival airport (e.g., Istanbul Airport). Copy this verbatim from the tool output - never infer or reuse a value from a different segment.")
    arrival_airport_code: Optional[str] = Field(default=None, description="The 3-letter IATA code (e.g., IST, SAW)")
    arrival_time: str = Field(description="Exact arrival time as returned by the tool, in the same format the tool provided it.")

# Instantiating the resonse schema based on the journey
class Journey(BaseModel):
    journey_type: Literal["Outbound", "Inbound"] = Field(description="Must be exactly 'Outbound' or 'Inbound'.")
    route: str = Field(description=("The full city-to-city route for THIS journey, listing every stop in order: "
                                    "origin city, then the city of every layover/connection, then the final destination city - "
                                    "joined with '-->'. You MUST use city names, NEVER full ariport names. The number of "
                                    "cities in this string must equal the number of segments plus one (a nonstop jornay has "
                                    "2 cities, e.g. 'Baku --> London'; a 1-stop journey has 3 e.g. 'Baku --> Istanbul --> London'; "
                                    "a 2-stop journey has 4, e.g. 'Baku --> Istanbul --> Budapest --> London'). NEVER collapse a "
                                    "multi-segment journey down to just the origin and final destination, and NEVER "
                                    "add a stop city that isn't backed by an actual segment."))
    total_duration: str = Field(description="The total duration of the journey, including layovers, as reported by the tool - do not estimate.")
    segments: List[FlightSegment] = Field(description="The individual physical flights that make up this journey, in travel order. You must contain at least one real segment grounded in tool data.")

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
    booking_link: Optional[str] = Field(default=None, description="URL for booking, copied exactly from the tool output. If the tool did not return a booking link, leave this empty - NEVER construct, guess, or template a URL yourself.")
    journeys: List[Journey] = Field(min_length=1, max_length=2, description="The directional journeys (e.g., Outbound and Inbound) that make up this flight option. Exactly one journey for a one-way search; exactly two (one Outbound, one Inbound) for a round-trip search.")

# Instantiating the response schema for all flights that meet the conditions
class FlightAgentResponse(BaseModel):
    found: bool = Field(description="True if the search tool returned at least one flight that genuinely satisfies the user's stated criteria (route, dates, stops, cabin class, etc). False if it did not — never set this True just to have something to return.")
    no_results_reason: Optional[str] = Field(default=None, description="Required when found=False, and omitted/null when found=True. A brief explanation, grounded in what the tool actually returned, of why no matching flight was found (e.g., 'No availability for GYD-LHR on 2026-09-20', 'All results had 3+ stops, more than the requested maximum of 2'). Do not guess at a reason that isn't supported by the tool output.")
    flight_options: List[FlightOption] = Field(default_factory=list, max_length=1, description="Exactly one flight option when found=True; must be left empty when found=False.")
    
    # Creating a model validator
    @model_validator(mode="after")
    def _found_is_consistent(self) -> "FlightAgentResponse":
        if self.found:
            if len(self.flight_options) != 1:
                raise ValueError(
                    "found=True requires exactly one entry in flight_options. If no flight "
                    "genuinely matched the user's criteria, set found=False and leave "
                    "flight_options empty instead of forcing an entry here."
                )
            if self.no_results_reason:
                raise ValueError("no_results_reason must be left empty/null when found=True.")
        else:
            if self.flight_options:
                raise ValueError(
                    "flight_options must be empty when found=False — do not include a fallback "
                    "or partial match here; explain the miss in no_results_reason instead."
                )
            if not self.no_results_reason:
                raise ValueError("no_results_reason is required and must be non-empty when found=False.")
        return self

# Defining an asynchronous function
async def build_flight_agent(llm: BaseChatModel, agent_tools: List[StructuredTool]) -> CompiledStateGraph:
    # Defining a system prompt
    flight_system_prompt="""
    You are a precise flight search expert.

    CRITICAL RULES:
        1. Base your response STRICTLY on the real data returned by your search tools. NEVER invent, guess, or hallucinate times, prices, airlines, or booking links. If a value is genuinely not present in the tool output, omit it (for optional fields) instead of inventing one — but omitting is only correct when the data truly isn't there. An optional field is exactly as wrong when you drop a value the tool actually provided as when you invent one it didn't. Check every optional field against the tool output for every segment before defaulting it to empty.
        2. TRIP SHAPE: A one-way request produces exactly ONE journey ('Outbound'). A round-trip request produces exactly TWO journeys ('Outbound' and 'Inbound'). Never add a return journey that was not requested, and never omit one that was.
        3. INBOUND GROUNDING: The Inbound journey is a DIFFERENT physical flight from the Outbound journey. It has its own departure/arrival airports (reversed direction from the Outbound), its own times, and often its own carrier and duration. NEVER derive the Inbound journey by copying, mirroring, or reversing the Outbound journey's data — always locate and parse the Inbound leg's own entry in the tool output.
        4. MISSING INBOUND DATA: If the user asked for a round trip but your most recent tool results only contain outbound data, DO NOT fabricate the inbound leg. Call the search tool again (explicitly for the return leg, if your tools are direction-specific) before writing your final answer. Only answer once you have real tool data for every requested direction.
        5. PREVENT DATA DUPLICATION: Process each flight option individually, one at a time — fully finish and double-check option 1 before starting option 2, and so on. Do not let a number from an earlier option influence a later one. Pay strict attention to exact prices, departure/arrival times, durations, and carriers, and specific airports (e.g., IST vs SAW). Two different flight options are ALLOWED to legitimately share an identical Outbound (or Inbound) journey (e.g., the same outbound flight combined with two different return flights) — that is not an error. What IS an error is writing a NEW itinerary with different departure/arrival times but then reusing the duration, carrier, or other descriptive fields from a different, similar-looking segment instead of reading them fresh from THIS segment's own entry in the tool output. If a route (e.g., via Istanbul) repeats across multiple flight options with different times, re-locate each option's own segment in the tool output by its specific departure_time before copying its duration and carrier — never assume they match another option just because the airports match.
        6. HIERARCHICAL PARSING: A 'FlightOption' consists of 'Journeys' (Outbound/Inbound). Each 'Journey' consists of one or more 'FlightSegments' (the actual physical flights). Accurately map layovers into multiple segments within a journey.
        7. MULTI-STOP ROUTES: If the user's search allows 1 or 2 stops and a journey actually has layovers, the 'route' field must list every city the journey passes through, in order — not just the origin and final destination. A 1-stop journey looks like 'Baku --> Istanbul --> London'; a 2-stop journey looks like 'Baku --> Istanbul --> Budapest --> London'. The number of cities in 'route' is always (number of segments + 1) for that journey. Never collapse a connecting itinerary down to a single origin-destination pair, and never list a stop that isn't backed by a real segment in the tool output.
        8. You MUST always use the default value if the user has not provided one for a particular search criterion.
        9. When executing the flight search tool, you MUST explicitly pass 'USD' as the currency parameter.
        10. Return exactly ONE flight option: the single best match for the user's stated criteria. If the user gave no preference beyond route, dates, and passengers, prefer the best balance of price and total duration. Do not include alternatives or a shortlist — one option only.
        11. NO MATCHING FLIGHT: If the search tool returns no results, or every result fails to satisfy a criterion the user actually stated (route, dates, max stops, cabin class, etc.), set found=False and explain why in no_results_reason, grounded in what the tool returned. NEVER relax a criterion the user specified in order to force a match, and NEVER invent a flight_option just so the response has one — an honest "no flight found" is always preferable to a fabricated ticket. Before concluding there are no results, make sure you've actually called the search tool (retrying with a broadened but still-reasonable query is fine, e.g. if the tool itself errors) — don't give up after a single failed call without checking the tool response.
        12. Strictly output the results conforming to the requested schema.

    WORKED EXAMPLES (for format only — never reuse these values):
        (a) Nonstop round trip Baku <-> London, tool results show two distinct itineraries:
        outbound: GYD 09:10 -> LHR 13:40, Azerbaijan Airlines
        return:   LHR 15:20 -> GYD 23:55, Azerbaijan Airlines
        Correct journeys = [
        {journey_type: "Outbound", route: "Baku --> London", segments: [GYD->LHR, 09:10-13:40]},
        {journey_type: "Inbound",  route: "London --> Baku", segments: [LHR->GYD, 15:20-23:55]}
        ]
        Note the Inbound segment's airports are reversed and its times are independent of the Outbound segment — never the same values restated.

        (b) One-stop outbound Baku -> London via Istanbul, tool results show two segments for the outbound leg:
        segment 1: GYD 09:10 -> IST 11:05, Turkish Airlines
        segment 2: IST 13:20 -> LHR 15:45, Turkish Airlines
        Correct journey = {
        journey_type: "Outbound",
        route: "Baku --> Istanbul --> London",
        segments: [GYD->IST 09:10-11:05, IST->LHR 13:20-15:45]
        }
        The route has 3 cities because there are 2 segments. A 2-stop itinerary (3 segments) would need 4 cities, e.g. "Baku --> Istanbul --> Budapest --> London".

    BEFORE YOU FINALIZE, VERIFY:
        - Every price, time, airport, duration, and airline you wrote appears in THIS segment's own entry in the tool output you actually received this turn — not a nearby segment's or a different flight option's entry.
        - The number of journeys matches the trip type (1 for one-way, 2 for round-trip).
        - If there are two journeys, the Inbound journey's route and segment times are genuinely different from the Outbound journey's (not copied or mirrored).
        - For every journey, 'route' lists exactly (number of segments + 1) cities, including every layover — not just the origin and final destination.
        - For every journey, 'total_duration' equals the sum of that journey's own segment durations plus the time spent at each layover (next segment's departure_time minus previous segment's arrival_time). If it doesn't add up, recompute it from this journey's own segments rather than leaving a copied value in place.
        - You did not invent a booking link — if the tool provided none, leave that field empty.
        - If found=False, flight_options is empty and no_results_reason is filled in; if found=True, flight_options has exactly one entry and no_results_reason is empty. Never a mix of the two.

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