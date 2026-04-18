import json
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Dict, List, Optional, TypedDict

from dotenv import load_dotenv
load_dotenv(dotenv_path=__file__.replace("buyer.py", ".env"))

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.types import Command, interrupt
from pydantic import BaseModel

from sell_buy_agents.seller_buyer_pool import find_sellers_for_buyer

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

MAX_ROUNDS = 5


# ---------------------------------------------------------------------------
# Buyer domain model
# ---------------------------------------------------------------------------

class BuyRequest(BaseModel):
    user_id: str
    user_name: str
    product_name: str
    min_price: float
    max_price: float
    city: str
    lat: float
    lon: float
    with_in_kms: float


class Buyer:
    def __init__(self, request: BuyRequest):
        self.user_id = request.user_id
        self.user_name = request.user_name
        self.product = request.product_name
        self.min_price = request.min_price
        self.max_price = request.max_price
        self.city = request.city
        self.lat = request.lat
        self.lon = request.lon
        self.with_in_kms = request.with_in_kms
        self.buy_status = "CREATED"
        self.buy_history: Dict[str, dict] = {}
        self.created_at = datetime.utcnow()
        self.closed_at: Optional[datetime] = None
        self.updated_at: Optional[datetime] = None

    def create_buy_request(self) -> str:
        self.buy_status = "PROCESSING"
        buy_request_id = str(uuid.uuid4())
        self.buy_history[buy_request_id] = {
            "buy_request_id": buy_request_id,
            "user_id": self.user_id,
            "user_name": self.user_name,
            "product_name": self.product,
            "min_price": self.min_price,
            "max_price": self.max_price,
            "city": self.city,
            "with_in_kms": self.with_in_kms,
            "buy_request_created": self.created_at.isoformat(),
            "buy_status": self.buy_status,
        }
        return buy_request_id

    def update_buy_request(self, buy_request_id: str,
                           min_price: float = None,
                           max_price: float = None,
                           status: str = None) -> None:
        buy_data = self.buy_history.get(buy_request_id)
        if buy_data:
            if min_price is not None:
                buy_data["min_price"] = min_price
            if max_price is not None:
                buy_data["max_price"] = max_price
            if status is not None:
                buy_data["buy_status"] = status
            self.updated_at = datetime.utcnow()

    def track_buy_history(self) -> Dict[str, dict]:
        return self.buy_history

    def close_buy_request(self, buy_request_id: str) -> None:
        self.buy_status = "CLOSED"
        self.closed_at = datetime.utcnow()
        buy_data = self.buy_history.get(buy_request_id)
        if buy_data:
            buy_data["closed_at"] = self.closed_at.isoformat()
            buy_data["buy_status"] = self.buy_status


# ---------------------------------------------------------------------------
# LangGraph Buy Agent
# ---------------------------------------------------------------------------

class State(TypedDict):
    user_id: str
    user_name: str
    product_name: str
    min_price: float
    max_price: float
    city: str
    lat: float
    lon: float
    with_in_kms: float
    response: str
    status: str
    history: List[dict]
    dialogs: List[dict]       # one entry per seller, populated by negotiate_all_sellers
    sellers_found: int
    buyers_found: int
    action: Optional[str]
    chosen_seller: Optional[dict]


# ---------------------------------------------------------------------------
# LLM helper
# ---------------------------------------------------------------------------

def _chat(system: str, conversation: List[dict]) -> dict:
    history_text = "\n".join(
        f"[{m['role'].upper()}]: {m['content']} (offer: ${m.get('offer', 'N/A')})"
        for m in conversation
    ) or "Start the negotiation with your opening offer."
    msgs = [
        SystemMessage(content=system + '\nReply with JSON only: {"offer": <number>, "message": "<text>", "accept": <bool>, "drop_off": <bool>}'),
        HumanMessage(content=history_text),
    ]
    resp = llm.invoke(msgs)
    try:
        return json.loads(resp.content)
    except json.JSONDecodeError:
        text = resp.content
        s, e = text.find("{"), text.rfind("}") + 1
        return json.loads(text[s:e]) if s != -1 else {"offer": 0.0, "message": text, "accept": False, "drop_off": False}


# ---------------------------------------------------------------------------
# Negotiation with a single seller (runs in a thread)
# ---------------------------------------------------------------------------

def _negotiate_with_one_seller(state: dict, seller: dict) -> dict:
    buyer_system = (
        f"You are a negotiation agent for buyer '{state['user_name']}' in {state['city']}.\n"
        f"Product: {state['product_name']}\n"
        f"Private price range: ${state['min_price']} – ${state['max_price']}\n"
        "Do NOT reveal your maximum price. Open near your min, concede slowly.\n"
        "Set accept=true if seller's offer fits your ceiling. Set drop_off=true if no deal is possible."
    )
    seller_system = (
        f"You are a negotiation agent for seller '{seller['user_name']}' in {seller['city']}.\n"
        f"Product: {seller['product_name']}\n"
        f"Private price range: ${seller['min_price']} – ${seller['max_price']}\n"
        "Do NOT reveal your minimum price. Open near your max, concede slowly. if buyer is not convinced, ask what the lowest price they are willing to buy\n"
        "Set accept=true if buyer's offer meets your floor."
    )

    conversation: List[dict] = []
    result_status = "ongoing"

    for _ in range(MAX_ROUNDS):
        b_data = _chat(buyer_system, conversation)
        conversation.append({"role": "buyer", "content": b_data["message"], "offer": b_data.get("offer")})
        if b_data.get("drop_off"):
            result_status = "drop_off"
            break

        s_data = _chat(seller_system, conversation)
        conversation.append({"role": "seller", "content": s_data["message"], "offer": s_data.get("offer")})
        if s_data.get("accept") or b_data.get("accept"):
            result_status = "agreed"
            break

    seller_offers = [m["offer"] for m in conversation if m["role"] == "seller" and m.get("offer") is not None]
    return {
        "sell_id": seller["sell_id"],
        "seller_name": seller["user_name"],
        "seller_city": seller["city"],
        "distance_km": seller.get("distance_km"),
        "conversation": conversation,
        "result": result_status,
        "best_seller_offer": seller_offers[-1] if seller_offers else None,
        "rounds": sum(1 for m in conversation if m["role"] == "buyer"),
    }


# ---------------------------------------------------------------------------
# Graph nodes
# ---------------------------------------------------------------------------

def get_sellers_numbers(state: State) -> State:
    matched = find_sellers_for_buyer({
        "product_name": state["product_name"],
        "lat": state["lat"], "lon": state["lon"],
        "with_in_kms": state["with_in_kms"],
    })
    count = len(matched)
    new_status = "HOLDING" if count == 0 else state.get("status", "PROCESSING")
    return {**state, "sellers_found": count, "status": new_status}


def categorize_buy_request(state: State) -> State:
    sellers = state.get("sellers_found", 0)
    buyers = state.get("buyers_found", 1)
    if sellers > buyers:
        response = "BUYER_MARKET: more sellers than buyers — negotiate hard"
    elif buyers > sellers:
        response = "SELLER_MARKET: few sellers — be flexible on price"
    else:
        response = "BALANCED: competitive pricing"
    history = state.get("history", []) + [{
        "event": "market_analysis", "time": datetime.utcnow().isoformat(), "response": response,
    }]
    return {**state, "response": response, "history": history}


def negotiate_all_sellers(state: State) -> State:
    """Concurrently negotiate with all matched sellers, collect dialogs into state."""
    matched_sellers = find_sellers_for_buyer({
        "product_name": state["product_name"],
        "lat": state["lat"], "lon": state["lon"],
        "with_in_kms": state["with_in_kms"],
    })

    if not matched_sellers:
        return {**state, "dialogs": [], "action": "reject"}

    print(f"\n  [BuyAgent] Negotiating concurrently with {len(matched_sellers)} seller(s)...\n")

    dialogs: List[dict] = [None] * len(matched_sellers)

    with ThreadPoolExecutor(max_workers=len(matched_sellers)) as executor:
        future_to_idx = {
            executor.submit(_negotiate_with_one_seller, state, seller): idx
            for idx, seller in enumerate(matched_sellers)
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            dialogs[idx] = future.result()

    # Print all conversations in order once all are done
    for d in dialogs:
        print(f"  ┌── Seller: {d['seller_name']} ({d['seller_city']})  [{d['distance_km']} km away]")
        for m in d["conversation"]:
            offer_str = f"  ${m['offer']}" if m.get("offer") is not None else ""
            print(f"  │  [{m['role'].upper():6}] {m['content']}{offer_str}")
        print(f"  └── Result: {d['result'].upper()}  |  Best seller offer: ${d['best_seller_offer']}\n")

    return {**state, "dialogs": dialogs}


def choose_seller(state: State) -> State:
    """LLM summarises all negotiations, then human picks the seller via interrupt."""
    dialogs = state.get("dialogs", [])

    if not dialogs:
        return {**state, "action": "reject", "chosen_seller": None}

    seller_lines = "\n".join(
        f"  Seller {i+1}: {d['seller_name']} ({d['seller_city']}) | "
        f"result={d['result']} | best offer=${d['best_seller_offer']} | rounds={d['rounds']}"
        for i, d in enumerate(dialogs)
    )
    summary_prompt = (
        f"Buyer '{state['user_name']}' wants to buy {state['product_name']} "
        f"for ${state['min_price']}–${state['max_price']}.\n\n"
        f"Concurrent negotiation outcomes:\n{seller_lines}\n\n"
        "In 3–4 sentences: summarise each seller's performance, recommend which seller "
        "the buyer should choose, and explain why (consider price offered, deal outcome, rounds taken)."
    )
    recommendation = llm.invoke([HumanMessage(content=summary_prompt)]).content

    options_str = "\n".join(
        f"    {i+1}. {d['seller_name']} ({d['seller_city']}) — "
        f"{d['result'].upper()} — best offer: ${d['best_seller_offer']}"
        for i, d in enumerate(dialogs)
    )
    interrupt_msg = (
        f"\n{'='*60}\n"
        f"  ALL NEGOTIATIONS COMPLETE\n"
        f"{'='*60}\n\n"
        f"  AI Recommendation:\n"
        + "\n".join(f"  {line}" for line in recommendation.splitlines()) +
        f"\n\n  Your sellers:\n{options_str}\n\n"
        f"  Enter number (1–{len(dialogs)}) or seller name to accept, "
        f"or 'skip' to pass on all:\n"
        f"{'='*60}"
    )

    raw_choice: str = interrupt(interrupt_msg)

    chosen = None
    raw = raw_choice.strip()
    if raw.lower() != "skip":
        try:
            idx = int(raw) - 1
            if 0 <= idx < len(dialogs):
                chosen = dialogs[idx]
        except ValueError:
            for d in dialogs:
                if d["seller_name"].lower() == raw.lower():
                    chosen = d
                    break

    if chosen:
        chosen["result"] = "accepted_by_buyer"
        print(f"\n  [BuyAgent] Deal accepted with {chosen['seller_name']}.")
    else:
        print("\n  [BuyAgent] No seller selected.")

    action = "accept" if chosen else "reject"
    return {**state, "action": action, "chosen_seller": chosen}


def hold_buy(state: State) -> str:
    return "end" if state.get("status") == "HOLDING" else "categorize"


def decide(state: State) -> str:
    action = state.get("action")
    if action == "accept":
        return "close"
    if action == "reject":
        return "search_more"
    return "wait"


def close(state: State) -> State:
    history = state.get("history", []) + [{"event": "closed", "time": datetime.utcnow().isoformat()}]
    return {**state, "status": "CLOSED", "history": history}


def save_memory(state: State) -> State:
    history = state.get("history", []) + [{"event": "saved_to_memory", "time": datetime.utcnow().isoformat()}]
    return {**state, "history": history}


# ---------------------------------------------------------------------------
# Buy Agent
# ---------------------------------------------------------------------------

class BuyAgent:
    """Each buyer user instantiates their own BuyAgent."""

    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self._app = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(State)

        graph.add_node("get_sellers_numbers", get_sellers_numbers)
        graph.add_node("categorize", categorize_buy_request)
        graph.add_node("negotiate_all_sellers", negotiate_all_sellers)
        graph.add_node("choose_seller", choose_seller)
        graph.add_node("close", close)
        graph.add_node("save_memory", save_memory)

        graph.set_entry_point("get_sellers_numbers")
        graph.add_conditional_edges("get_sellers_numbers", hold_buy, {
            "end": END,
            "categorize": "categorize",
        })
        graph.add_edge("categorize", "negotiate_all_sellers")
        graph.add_edge("negotiate_all_sellers", "choose_seller")
        graph.add_edge("choose_seller", "save_memory")
        graph.add_edge("close", END)
        graph.add_conditional_edges("save_memory", decide, {
            "close": "close",
            "wait": END,
            "search_more": "negotiate_all_sellers",
        })

        return graph.compile(checkpointer=MemorySaver())

    def run(self, buyer: "Buyer", buy_request_id: str) -> dict:
        config = {"configurable": {"thread_id": f"{self.agent_id}-{buy_request_id}"}}
        initial_state: State = {
            "user_id": buyer.user_id,
            "user_name": buyer.user_name,
            "product_name": buyer.product,
            "min_price": buyer.min_price,
            "max_price": buyer.max_price,
            "city": buyer.city,
            "lat": buyer.lat,
            "lon": buyer.lon,
            "with_in_kms": buyer.with_in_kms,
            "response": "",
            "status": "PROCESSING",
            "history": [],
            "dialogs": [],
            "sellers_found": 0,
            "buyers_found": 1,
            "action": None,
            "chosen_seller": None,
        }

        result = {}
        for chunk in self._app.stream(initial_state, config, stream_mode="values"):
            result = chunk

        while result.get("__interrupt__"):
            print(result["__interrupt__"][0].value)
            human_input = input("  >>> Your choice: ").strip()
            result = {}
            for chunk in self._app.stream(Command(resume=human_input), config, stream_mode="values"):
                result = chunk

        return result
