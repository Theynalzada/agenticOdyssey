# Importing Dependencies
from langchain_core.language_models import BaseChatModel
from langchain_mcp_adapters.tools import StructuredTool
from langgraph.graph.state import CompiledStateGraph
from langchain.messages import SystemMessage
from typing import List, Literal, Optional
from langchain.agents import create_agent
from pydantic import Field, BaseModel

# Defining a response format
class VisaAgentResponse(BaseModel):
  user_nationality: str = Field(description="The user's nationality expressed as a demonym/adjective (e.g. 'Indian', 'Azerbaijani'), never the country name itself (e.g. not 'India').")
  destination_country: str = Field(description="The country the user wants to visit.")
  visa_type: Literal["tourist", "business", "transit"] = Field(description="The type of visa relevant to the user's trip. Default to 'tourist' unless the user's query indicates otherwise.")
  visa_status: Literal["visa_free", "visa_on_arrival", "e_visa", "visa_required"] = Field(description="The visa category that applies for this nationality/destination pair.")
  stay_duration_days: Optional[int] = Field(default=None, description="The maximum number of days the user is permitted to stay under this visa status or visa type, if specified by the source material.")
  requirements: Optional[List[str]] = Field(default=None, description="The list of documents and requirements needed to apply for the visa, applicable to any destination country. Leave empty if visa_status is 'visa_free'.")
  schengen_countries: Optional[List[str]] = Field(default=None, description="Only relevant when this nationality requires a visa for the destination (visa_status is not 'visa_free'). The Schengen member countries whose visa would also satisfy entry into the destination, if a Schengen visa issued by another member state can be used instead of the destination's own visa — either because the destination is itself a Schengen member (any Schengen visa works area-wide) or because it has a policy accepting valid Schengen visas in lieu of its own. Leave empty whenever visa_status is 'visa_free', or when no Schengen-visa alternative exists.")
  source: Optional[List[str]] = Field(default=None, description="The list of URLs returned by search tools that were used to determine this information.")

# Defining an asynchronous function
async def build_visa_agent(llm: BaseChatModel, agent_tools: List[StructuredTool]) -> CompiledStateGraph:
  # Defining a system prompt
  visa_system_prompt="""
  You are a visa expert. You are extremely capable of finding detailed,
  up-to-date information about the visa requirements for a specific
  nationality traveling to a specific destination country.
  Focus your search and response on this specific nationality/destination pair.
  Your answer must be precise regarding requirements and your tone must be professional.
  You MUST strictly output the results conforming to the requested schema.
 
  FIELD LOGIC:
  - Determine visa_status first, then populate every other field consistently with it.
  - If visa_status is "visa_free" — meaning this nationality can enter the destination without
    ANY visa at all, Schengen or otherwise — leave BOTH requirements and schengen_countries
    empty. There is no visa to obtain and therefore no alternate Schengen-visa route to report.
  - Only when visa_status is "visa_on_arrival", "e_visa", or "visa_required" (i.e. this
    nationality does need some visa to enter) should you evaluate schengen_countries:
      (a) if the destination is itself a Schengen member state, list the other Schengen member
          states, since a Schengen visa issued by any of them is valid area-wide and can be used
          instead of a visa from the destination specifically; or
      (b) if the destination is outside the Schengen area but has a specific policy accepting
          valid Schengen visas in place of its own visa (e.g. Georgia's waiver for Schengen-visa
          holders), list those Schengen states.
      Otherwise (no Schengen-visa alternative exists), leave schengen_countries empty.
  - If visa_status is "visa_on_arrival", "e_visa", or "visa_required", populate requirements
    with the concrete documents and steps needed (e.g. passport validity, application form,
    proof of funds, application fee, processing time), based on what you find via search.
  - Always express user_nationality as a demonym (e.g. "Indian", "Azerbaijani"), never as the
    country name (e.g. not "India").
  - When schengen_countries is populated, list actual Schengen member country names (e.g.
    "France", "Germany", "Italy") — never the umbrella phrase "Schengen Area" or similar.
  - Populate stay_duration_days only if a specific number of days is stated by your sources.
  - Populate source with the actual URLs returned by your search tool calls. Do not fabricate URLs.
  - Visa rules change frequently. Note in your search that you are looking for the most current
    official guidance available, and prefer official government/embassy sources when present.
 
  CRITICAL RULE: Populate the final output with the actual scraped visa data.
  Never output JSON schema definitions, field types, or descriptions inside the values.
  """
  
  # Creating a system message
  visa_system_message=SystemMessage(content=visa_system_prompt)

  # Defining an agent that checks visa availability
  visa_agent=create_agent(model=llm,
                          tools=agent_tools,
                          system_prompt=visa_system_message,
                          response_format=VisaAgentResponse,
                          name="visa_agent")
  
  # Returning the agent
  return visa_agent