import json
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Dict, List, Optional, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.types import Command, interrupt
from pydantic import BaseModel

from sell_buy_agents.seller_buyer_pool import find_buyers_for_seller
from dotenv import load_dotenv
load_dotenv(dotenv_path=__file__.replace("seller.py", ".env"))
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

MAX_ROUNDS = 10


# ---------------------------------------------------------------------------
# Seller domain model
# ---------------------------------------------------------------------------

class SellRequest(BaseModel):
    user_id: str
    user_name: str
    product_name: str
    min_price: float
    max_price: float
    city: str
    lat: float
    lon: float
    with_in_kms: float


class Seller:
    def __init__(self, request: SellRequest):
        self.user_id = request.user_id
        self.user_name = request.user_name
        self.product = request.product_name
        self.min_price = request.min_price
        self.max_price = request.max_price
        self.city = request.city
        self.lat = request.lat
        self.lon = request.lon
        self.with_in_kms = request.with_in_kms
        self.sell_status = "CREATED"
        self.sell_history: Dict[str, dict] = {}
        self.created_at = datetime.utcnow()
        self.closed_at: Optional[datetime] = None
        self.updated_at: Optional[datetime] = None

    def create_sell_request(self) -> str:
        self.sell_status = "PROCESSING"
        sell_request_id = str(uuid.uuid4())
        self.sell_history[sell_request_id] = {
            "sell_request_id": sell_request_id,
            "user_id": self.user_id,
            "user_name": self.user_name,
            "product_name": self.product,
            "min_price": self.min_price,
            "max_price": self.max_price,
            "city": self.city,
            "with_in_kms": self.with_in_kms,
            "sell_request_created": self.created_at.isoformat(),
            "sell_status": self.sell_status,
        }
        return sell_request_id

    def update_sell_request(self, sell_request_id: str,
                            min_price: float = None,
                            max_price: float = None,
                            status: str = None) -> None:
        sell_data = self.sell_history.get(sell_request_id)
        if sell_data:
            if min_price is not None:
                sell_data["min_price"] = min_price
            if max_price is not None:
                sell_data["max_price"] = max_price
            if status is not None:
                sell_data["sell_status"] = status
            self.updated_at = datetime.utcnow()

    def track_sell_history(self) -> Dict[str, dict]:
        return self.sell_history

    def close_sell_request(self, sell_request_id: str) -> None:
        self.sell_status = "CLOSED"
        self.closed_at = datetime.utcnow()
        sell_data = self.sell_history.get(sell_request_id)
        if sell_data:
            sell_data["closed_at"] = self.closed_at.isoformat()
            sell_data["sell_status"] = self.sell_status


# ---------------------------------------------------------------------------
# LangGraph Sell Agent
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
    dialogs: List[dict]       # one entry per buyer, populated by negotiate_all_buyers
    buyers_found: int
    sellers_found: int
    action: Optional[str]
    chosen_buyer: Optional[dict]


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
# Negotiation with a single buyer (runs in a thread)
# ---------------------------------------------------------------------------

def _negotiate_with_one_buyer(state: dict, buyer: dict) -> dict:
    seller_system = (
        f"You are a negotiation agent for seller '{state['user_name']}' in {state['city']}.\n"
        f"Product: {state['product_name']}\n"
        f"Private price range: ${state['min_price']} – ${state['max_price']}\n"
        "Do NOT reveal your minimum price. Open near your max, concede slowly.\n"
        "Set accept=true if buyer's offer meets your floor. Set drop_off=true if no deal is possible."
    )
    buyer_system = (
        f"You are a negotiation agent for buyer '{buyer['user_name']}' in {buyer['city']}.\n"
        f"Product: {buyer['product_name']}\n"
        f"Private price range: ${buyer['min_price']} – ${buyer['max_price']}\n"
        "Do NOT reveal your maximum price. Open near your min, concede slowly.\n"
        "Set accept=true if seller's offer fits your ceiling."
    )

    conversation: List[dict] = []
    result_status = "ongoing"

    for _ in range(MAX_ROUNDS):
        s_data = _chat(seller_system, conversation)
        conversation.append({"role": "seller", "content": s_data["message"], "offer": s_data.get("offer")})
        if s_data.get("drop_off"):
            result_status = "drop_off"
            break

        b_data = _chat(buyer_system, conversation)
        conversation.append({"role": "buyer", "content": b_data["message"], "offer": b_data.get("offer")})
        if b_data.get("accept") or s_data.get("accept"):
            result_status = "agreed"
            break

    buyer_offers = [m["offer"] for m in conversation if m["role"] == "buyer" and m.get("offer") is not None]
    return {
        "buyer_id": buyer["buyer_id"],
        "buyer_name": buyer["user_name"],
        "buyer_city": buyer["city"],
        "distance_km": buyer.get("distance_km"),
        "conversation": conversation,
        "result": result_status,
        "best_buyer_offer": buyer_offers[-1] if buyer_offers else None,
        "rounds": sum(1 for m in conversation if m["role"] == "seller"),
    }


# ---------------------------------------------------------------------------
# Graph nodes
# ---------------------------------------------------------------------------

def get_buyers_numbers(state: State) -> State:
    matched = find_buyers_for_seller({
        "product_name": state["product_name"],
        "lat": state["lat"], "lon": state["lon"],
        "with_in_kms": state["with_in_kms"],
    })
    count = len(matched)
    new_status = "HOLDING" if count == 0 else state.get("status", "PROCESSING")
    return {**state, "buyers_found": count, "status": new_status}


def categorize_sell_request(state: State) -> State:
    buyers = state.get("buyers_found", 0)
    sellers = state.get("sellers_found", 1)
    if sellers > buyers:
        response = "BUYER_MARKET: consider lowering your price"
    elif buyers > sellers:
        response = "SELLER_MARKET: you can hold a higher price"
    else:
        response = "BALANCED: competitive pricing"
    history = state.get("history", []) + [{
        "event": "market_analysis", "time": datetime.utcnow().isoformat(), "response": response,
    }]
    return {**state, "response": response, "history": history}


def negotiate_all_buyers(state: State) -> State:
    """Concurrently negotiate with all matched buyers, collect dialogs into state."""
    matched_buyers = find_buyers_for_seller({
        "product_name": state["product_name"],
        "lat": state["lat"], "lon": state["lon"],
        "with_in_kms": state["with_in_kms"],
    })

    if not matched_buyers:
        return {**state, "dialogs": [], "action": "reject"}

    print(f"\n  [SellAgent] Negotiating concurrently with {len(matched_buyers)} buyer(s)...\n")

    dialogs: List[dict] = [None] * len(matched_buyers)

    with ThreadPoolExecutor(max_workers=len(matched_buyers)) as executor:
        future_to_idx = {
            executor.submit(_negotiate_with_one_buyer, state, buyer): idx
            for idx, buyer in enumerate(matched_buyers)
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            dialogs[idx] = future.result()

    # Print all conversations in order once all are done
    for d in dialogs:
        print(f"  ┌── Buyer: {d['buyer_name']} ({d['buyer_city']})  [{d['distance_km']} km away]")
        for m in d["conversation"]:
            sleep(0.1)
            offer_str = f"  ${m['offer']}" if m.get("offer") is not None else ""
            print(f"  │  [{m['role'].upper():6}] {m['content']}{offer_str}")
            sleep(0.1)
        print(f"  └── Result: {d['result'].upper()}  |  Best offer: ${d['best_buyer_offer']}\n")

    return {**state, "dialogs": dialogs}


def choose_buyer(state: State) -> State:
    """LLM summarises all negotiations, then human picks the buyer via interrupt."""
    dialogs = state.get("dialogs", [])

    if not dialogs:
        return {**state, "action": "reject", "chosen_buyer": None}

    # Build LLM summary
    buyer_lines = "\n".join(
        f"  Buyer {i+1}: {d['buyer_name']} ({d['buyer_city']}) | "
        f"result={d['result']} | best offer=${d['best_buyer_offer']} | rounds={d['rounds']}"
        for i, d in enumerate(dialogs)
    )
    summary_prompt = (
        f"Seller '{state['user_name']}' wants to sell {state['product_name']} "
        f"for ${state['min_price']}–${state['max_price']}.\n\n"
        f"Concurrent negotiation outcomes:\n{buyer_lines}\n\n"
        "In 3–4 sentences: summarise each buyer's performance, recommend which buyer "
        "the seller should choose, and explain why (consider price offered, deal outcome, rounds taken)."
    )
    recommendation = llm.invoke([HumanMessage(content=summary_prompt)]).content

    # Build the interrupt message the seller will see
    options_str = "\n".join(
        f"    {i+1}. {d['buyer_name']} ({d['buyer_city']}) — "
        f"{d['result'].upper()} — best offer: ${d['best_buyer_offer']}"
        for i, d in enumerate(dialogs)
    )
    interrupt_msg = (
        f"\n{'='*60}\n"
        f"  ALL NEGOTIATIONS COMPLETE\n"
        f"{'='*60}\n\n"
        f"  AI Recommendation:\n"
        + "\n".join(f"  {line}" for line in recommendation.splitlines()) +
        f"\n\n  Your buyers:\n{options_str}\n\n"
        f"  Enter number (1–{len(dialogs)}) or buyer name to accept, "
        f"or 'skip' to pass on all:\n"
        f"{'='*60}"
    )

    raw_choice: str = interrupt(interrupt_msg)

    # Resolve choice to a dialog entry
    chosen = None
    raw = raw_choice.strip()
    if raw.lower() != "skip":
        try:
            idx = int(raw) - 1
            if 0 <= idx < len(dialogs):
                chosen = dialogs[idx]
        except ValueError:
            for d in dialogs:
                if d["buyer_name"].lower() == raw.lower():
                    chosen = d
                    break

    if chosen:
        chosen["result"] = "accepted_by_seller"
        print(f"\n  [SellAgent] Deal accepted with {chosen['buyer_name']}.")
    else:
        print("\n  [SellAgent] No buyer selected.")

    action = "accept" if chosen else "reject"
    return {**state, "action": action, "chosen_buyer": chosen}


def hold_sell(state: State) -> str:
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
# Sell Agent
# ---------------------------------------------------------------------------

class SellAgent:
    """Each seller user instantiates their own SellAgent."""

    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self._app = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(State)

        graph.add_node("get_buyers_numbers", get_buyers_numbers)
        graph.add_node("categorize", categorize_sell_request)
        graph.add_node("negotiate_all_buyers", negotiate_all_buyers)
        graph.add_node("choose_buyer", choose_buyer)
        graph.add_node("close", close)
        graph.add_node("save_memory", save_memory)

        graph.set_entry_point("get_buyers_numbers")
        graph.add_conditional_edges("get_buyers_numbers", hold_sell, {
            "end": END,
            "categorize": "categorize",
        })
        graph.add_edge("categorize", "negotiate_all_buyers")
        graph.add_edge("negotiate_all_buyers", "choose_buyer")
        graph.add_edge("choose_buyer", "save_memory")
        graph.add_edge("close", END)
        graph.add_conditional_edges("save_memory", decide, {
            "close": "close",
            "wait": END,
            "search_more": "negotiate_all_buyers",
        })

        return graph.compile(checkpointer=MemorySaver())

    def run(self, seller: "Seller", sell_request_id: str) -> dict:
        config = {"configurable": {"thread_id": f"{self.agent_id}-{sell_request_id}"}}
        initial_state: State = {
            "user_id": seller.user_id,
            "user_name": seller.user_name,
            "product_name": seller.product,
            "min_price": seller.min_price,
            "max_price": seller.max_price,
            "city": seller.city,
            "lat": seller.lat,
            "lon": seller.lon,
            "with_in_kms": seller.with_in_kms,
            "response": "",
            "status": "PROCESSING",
            "history": [],
            "dialogs": [],
            "buyers_found": 0,
            "sellers_found": 1,
            "action": None,
            "chosen_buyer": None,
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
