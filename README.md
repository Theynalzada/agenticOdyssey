# Agentic Odyssey ✈️

**Agentic Odyssey** is a multi-agent AI travel planner built with [LangGraph](https://github.com/langchain-ai/langgraph) and [LangChain](https://github.com/langchain-ai/langchain). A **supervisor agent** holds a conversation with the user, incrementally collects trip details, and delegates specialized sub-tasks to three worker agents — **visa**, **hotel**, and **flight** — each of which returns strictly-typed, schema-validated results. The whole system runs as a single LangGraph graph that can be served locally with the LangGraph API/Studio or driven directly from a notebook.

> ⚠️ This repository has no license file and no formal releases yet — treat it as a personal/experimental project rather than production-ready software.

---

## Table of Contents

- [How it works](#how-it-works)
- [Key features](#key-features)
- [Architecture](#architecture)
- [Repository structure](#repository-structure)
- [Tech stack](#tech-stack)
- [Prerequisites](#prerequisites)
- [Setup](#setup)
- [Running the agent](#running-the-agent)
  - [Option A — LangGraph dev server / Studio](#option-a--langgraph-dev-server--studio)
  - [Option B — Jupyter notebook](#option-b--jupyter-notebook)
- [Example conversation](#example-conversation)
- [Conversation state](#conversation-state)
- [Response schema](#response-schema)
- [Sub-agents in detail](#sub-agents-in-detail)
- [MCP tool servers](#mcp-tool-servers)

---

## How it works

A user describes a trip in natural language (destination, dates, nationality, passenger count, preferences, etc.), possibly spread across several messages. The **supervisor agent**:

1. Extracts whatever trip details are present in each message and persists them into a shared conversation state via an `update_state` tool.
2. As soon as the prerequisites for a given specialist are satisfied, **proactively** (without being explicitly asked) calls that specialist:
   - `consult_visa_agent` — once destination country + user's nationality are known.
   - `consult_hotel_agent` — once destination city is known.
   - `consult_flight_agent` — once origin city, destination city, and departure date are known.
3. Merges the specialists' structured outputs into a single response object, only adding conversational text (`message_to_user`) for things the structured data can't say — e.g. asking a clarifying question, or a short wrap-up note.
4. Responds to the user in whatever language they're conversing in (a `dynamic_prompt` middleware injects a language instruction based on runtime context).

Conversation memory is checkpointed in-process (`InMemorySaver`) and automatically summarized once it grows past 75% of the context budget, retaining the most recent 25% verbatim (`SummarizationMiddleware`), so long multi-turn trip-planning sessions don't blow the context window.

## Key features

- **Supervisor / sub-agent architecture** — one orchestrating agent, three specialist agents, each with its own system prompt, tools, and Pydantic response schema.
- **Incremental slot-filling** — trip details can be provided over multiple turns; the supervisor tracks what's known and what's still missing instead of re-asking or guessing.
- **Proactive delegation** — specialists are consulted automatically the moment their required inputs are available, rather than waiting for an explicit user request per topic.
- **Strict structured outputs** — every agent (visa, hotel, flight, and the supervisor itself) returns a validated Pydantic model, not free text, so downstream code/UIs can rely on the shape of the data.
- **Multilingual by design** — a `SupervisorContext.user_language` value is threaded through a dynamic prompt so the entire conversation (including sub-agent outputs like visa requirements) can be produced in the user's language.
- **Real, live data only** — every agent's system prompt explicitly forbids inventing or hallucinating booking links, hotel data, or visa rules; results must come from actual tool calls.
- **Guardrails against crashes** — each `consult_*` tool checks its prerequisites and returns a structured `{"error": ...}` payload (rather than throwing) if required state is missing, so the supervisor can ask a follow-up question instead of failing.
- **Conversation summarization** — automatic trimming/summarizing of long chat histories to stay within model context limits.
- **MCP-powered tools** — flight search and web search are provided by external Model Context Protocol (MCP) servers rather than hand-rolled API integrations.

## Architecture

```
                          ┌─────────────────────────┐
                          │   Supervisor Agent       │
                          │  (gpt-5-nano, LangGraph) │
                          │                          │
User ── conversation ───▶│  tools:                  │
                          │   • update_state         │
                          │   • consult_visa_agent   │──────┐
                          │   • consult_hotel_agent  │──┐   │
                          │   • consult_flight_agent │──┼─┐ │
                          └─────────────────────────┘  │ │ │
                                                        │ │ │
                          ┌─────────────────────────┐   │ │ │
                          │      Flight Agent        │◀──┘ │ │
                          │  tools: Kiwi "search-    │     │ │
                          │  flight" (MCP)           │     │ │
                          └─────────────────────────┘     │ │
                                                           │ │
                          ┌─────────────────────────┐     │ │
                          │      Hotel Agent         │◀────┘ │
                          │  tools: Tavily web       │       │
                          │  search (MCP)            │       │
                          └─────────────────────────┘       │
                                                             │
                          ┌─────────────────────────┐       │
                          │      Visa Agent          │◀──────┘
                          │  tools: Tavily web       │
                          │  search (MCP)            │
                          └─────────────────────────┘
```

Each sub-agent is itself a fully independent `create_agent(...)` graph (from `langchain.agents`) with its own system prompt, tool set, and `response_format`; the supervisor invokes them as tools (`ainvoke`) and passes through their structured output unmodified.

## Repository structure

```
agenticOdyssey/
├── Notebooks/
│   └── playground.ipynb      # Interactive scratchpad demonstrating an end-to-end run
├── Scripts/
│   ├── __init__.py
│   ├── travel_planner.py     # Supervisor agent: state, tools, prompt, graph assembly
│   ├── visa_agent.py         # Visa specialist agent + VisaAgentResponse schema
│   ├── hotel_agent.py        # Hotel specialist agent + HotelAgentResponse schema
│   └── flight_agent.py       # Flight specialist agent + FlightAgentResponse schema
├── langgraph.json            # LangGraph CLI/Studio graph configuration
├── requirements.txt          # Python dependencies
└── .gitignore
```

*(A `.langgraph_api/` directory may also appear locally — it's created by the LangGraph dev server to store local run checkpoints/state and isn't meant to be committed.)*

## Tech stack

| Component | Purpose |
|---|---|
| [LangGraph](https://github.com/langchain-ai/langgraph) `1.2.11` | Graph runtime, state management, checkpointing (`InMemorySaver`), `Command` updates |
| [LangChain](https://github.com/langchain-ai/langchain) `1.4.0` | `create_agent`, middleware (`dynamic_prompt`, `SummarizationMiddleware`), chat model init |
| [langchain-mcp-adapters](https://pypi.org/project/langchain-mcp-adapters/) `0.3.2` | Bridges MCP servers' tools into LangChain `StructuredTool`s |
| [Pydantic](https://docs.pydantic.dev/) `2.13.5` | Schemas for every agent's structured response |
| `python-dotenv` `1.2.3` | Loads secrets from a local `.env` file |
| **gpt-5-nano** (via `init_chat_model`) | The LLM backing every agent in the graph |
| **Tavily MCP** (`tavily-mcp` via `npx`) | Web search used by the hotel and visa agents |
| **Kiwi MCP** (`https://mcp.kiwi.com`) | Flight search (`search-flight` tool) used by the flight agent |

## Prerequisites

- **Python 3.11+** (recommended, for compatibility with current LangChain/LangGraph releases)
- **Node.js + npx** — the Tavily MCP server is launched on demand via `npx -y tavily-mcp`
- **An OpenAI API key** — `gpt-5-nano` is resolved through LangChain's `init_chat_model`, which expects `OPENAI_API_KEY` in the environment
- **A Tavily API key** — for the web-search MCP server used by the hotel and visa agents
- Outbound network access to `https://mcp.kiwi.com` (the flight-search MCP server; no key is wired up for it in this codebase)
- (Optional) [LangGraph CLI](https://langchain-ai.github.io/langgraph/cloud/reference/cli/) if you want to run the graph via `langgraph dev` / LangGraph Studio, as `langgraph.json` is configured for

## Setup

```bash
# 1. Clone the repository
git clone https://github.com/Theynalzada/agenticOdyssey.git
cd agenticOdyssey

# 2. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 3. Install Python dependencies
pip install -r requirements.txt

# 4. (If you plan to use the LangGraph dev server/Studio)
pip install -U "langgraph-cli[inmem]"
```

Create a `.env` file in the project root (this is loaded by `load_dotenv()` in `travel_planner.py` and referenced by `langgraph.json`):

```bash
OPENAI_API_KEY=sk-...
TAVILY_API_KEY=tvly-...
```

## Running the agent

### Option A — LangGraph dev server / Studio

`langgraph.json` already points at the graph's entry point (`build_supervisor_agent` in `Scripts/travel_planner.py`), so you can launch it directly:

```bash
langgraph dev
```

This starts a local API server and opens LangGraph Studio in the browser, where you can chat with the `agent` graph interactively, inspect state, and step through tool calls.

### Option B — Jupyter notebook

`Notebooks/playground.ipynb` shows an end-to-end scripted example:

```python
from langchain.messages import HumanMessage
from Scripts.travel_planner import build_supervisor_agent, SupervisorContext

memory_config = {"configurable": {"thread_id": "1"}}
planner_agent = await build_supervisor_agent()

response = await planner_agent.ainvoke(
    input={"messages": [HumanMessage(content="Plan my trip to Rome...")]},
    config=memory_config,
    context=SupervisorContext(user_language="English"),
)

response["structured_response"]
```

Run Jupyter from the repo root (or adjust `sys.path` as the notebook does) so the `Scripts` package resolves correctly.

## Example conversation

Input used in `playground.ipynb`:

> *As a citizen of Azerbaijan departing from Baku, plan a 5-day trip to Rome, Italy. I am travelling with my friend and both of us have 1 checked baggage each. This time we have chosen business as the cabin class. Departure date: 2026-10-20, Return date: 2026-10-25.*

The supervisor extracted every detail in one shot (destination, origin, dates, cabin class, passenger/baggage counts) via `update_state`, then automatically consulted all three specialists and returned:

- **Visa info** — status (`visa_required`), a full list of required documents, and the set of Schengen countries whose visa would also grant entry.
- **Hotel info** — a list of 5-star hotels in Rome with name, address, city, and country.
- **Flight info** — an ordered list of booking links for matching business-class tickets.
- **A short wrap-up message** in the requested language, without repeating the structured data.

## Conversation state

`SupervisorState` (extends LangGraph's `AgentState`) tracks the trip's slots across turns:

| Field | Description |
|---|---|
| `destination_country`, `countryFrom` | Destination and the user's nationality — required for visa lookup |
| `destination_city`, `cityFrom` | Destination and origin cities — required for hotel/flight lookup |
| `departure_date`, `return_date` | Trip dates |
| `preferred_airline`, `cabin_class` | Flight preferences |
| `n_hotels`, `stars` | Hotel search preferences |
| `n_adults`, `n_children`, `n_infants`, `n_checked_baggage` | Passenger/baggage counts |

Fields are only overwritten when the user actually supplies a new value — the `update_state` tool merges rather than resets.

## Response schema

Every supervisor turn returns a `SupervisorAgentResponse`:

```python
class SupervisorAgentResponse(BaseModel):
    message_to_user: Optional[str]
    visa_info: Optional[VisaAgentResponse]
    hotel_info: Optional[HotelAgentResponse]
    flight_info: Optional[FlightAgentResponse]
```

Each `*_info` field is left as `None` until the corresponding specialist has actually been consulted for the current trip.

## Sub-agents in detail

### Visa agent (`Scripts/visa_agent.py`)
Determines `visa_status` (`visa_free` / `visa_on_arrival` / `e_visa` / `visa_required`) for a nationality → destination pair, and conditionally populates required documents, permitted stay duration, and any Schengen-visa alternative route — all sourced from live web search (Tavily), with source URLs included.

### Hotel agent (`Scripts/hotel_agent.py`)
Searches for hotels matching a city and star rating, returning structured `name`, `address`, `city`, `country`, and `stars` for each, de-duplicating repeated results.

### Flight agent (`Scripts/flight_agent.py`)
Searches for flights via the Kiwi MCP `search-flight` tool, carefully mapping user-facing preferences (cabin class letters, IATA airline codes, per-passenger baggage arrays, `dd/mm/yyyy` dates, USD currency, price-ascending sort) into the tool's exact parameter format, then re-verifies results against every explicitly stated constraint before returning booking links.

## MCP tool servers

The supervisor wires up an `MultiServerMCPClient` with two servers:

| Server | Transport | Used by | Notes |
|---|---|---|---|
| `tavily-mcp` | stdio (`npx -y tavily-mcp`) | Hotel agent, Visa agent | Requires `TAVILY_API_KEY` |
| `kiwi` | streamable-http (`https://mcp.kiwi.com`) | Flight agent | Public Kiwi.com flight search endpoint |
