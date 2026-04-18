"""
FastAPI backend for the SellBuy personal portal.

Start:
    cd /workspaces/SellBuy
    uvicorn playground.server:app --reload --port 8000
"""

import asyncio
import json
import sys
import threading
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / "sell_buy_agents" / ".env")

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from sell_buy_agents.seller_buyer_pool import (
    BUYERS, SELLERS,
    find_buyers_for_seller, find_sellers_for_buyer,
)

llm   = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
MAX_ROUNDS = 5
app   = FastAPI()

# ── In-memory "DB" ────────────────────────────────────────────────────────────
requests_db: dict[str, dict] = {}   # request_id → request
ws_sessions:  dict[str, dict] = {}   # request_id → {interrupt_event, interrupt_resp}

# ── Pre-seeded pool (background agents other users can match against) ─────────
SELLERS.extend([
    {"sell_id":"seed-S1","user_id":"seed","user_name":"Alice",  "product_name":"iPhone 15",  "city":"San Francisco","lat":37.7749,"lon":-122.4194,"min_price":750, "max_price":950, "with_in_kms":15},
    {"sell_id":"seed-S2","user_id":"seed","user_name":"Charlie","product_name":"MacBook Pro", "city":"San Jose",     "lat":37.3382,"lon":-121.8863,"min_price":1200,"max_price":1600,"with_in_kms":15},
    {"sell_id":"seed-S3","user_id":"seed","user_name":"Eve",    "product_name":"iPad Air",   "city":"Oakland",      "lat":37.8044,"lon":-122.2712,"min_price":400, "max_price":600, "with_in_kms":15},
    {"sell_id":"seed-S4","user_id":"seed","user_name":"Grace",  "product_name":"AirPods Pro","city":"Sunnyvale",    "lat":37.3688,"lon":-122.0363,"min_price":150, "max_price":250, "with_in_kms":15},
    {"sell_id":"seed-S5","user_id":"seed","user_name":"Ivan",   "product_name":"Samsung TV", "city":"San Jose",     "lat":37.3382,"lon":-121.8863,"min_price":400, "max_price":700, "with_in_kms":15},
])
BUYERS.extend([
    {"buyer_id":"seed-B1","user_id":"seed","user_name":"Bob",   "product_name":"iPhone 15",  "city":"Oakland",      "lat":37.8044,"lon":-122.2712,"min_price":600, "max_price":860, "with_in_kms":15},
    {"buyer_id":"seed-B2","user_id":"seed","user_name":"Diana", "product_name":"MacBook Pro", "city":"Santa Clara",  "lat":37.3541,"lon":-121.9552,"min_price":1000,"max_price":1500,"with_in_kms":15},
    {"buyer_id":"seed-B3","user_id":"seed","user_name":"Frank", "product_name":"iPad Air",   "city":"Berkeley",     "lat":37.8716,"lon":-122.2727,"min_price":300, "max_price":550, "with_in_kms":15},
    {"buyer_id":"seed-B4","user_id":"seed","user_name":"Henry", "product_name":"AirPods Pro","city":"Mountain View","lat":37.3861,"lon":-122.0839,"min_price":100, "max_price":200, "with_in_kms":15},
    {"buyer_id":"seed-B5","user_id":"seed","user_name":"Judy",  "product_name":"Samsung TV", "city":"Sunnyvale",    "lat":37.3688,"lon":-122.0363,"min_price":300, "max_price":600, "with_in_kms":15},
])


# ── REST: requests ─────────────────────────────────────────────────────────────

class RequestBody(BaseModel):
    user_id: str
    user_name: str
    role: str
    product_name: str
    min_price: float
    max_price: float
    city: str
    lat: float
    lon: float
    with_in_kms: float


def _update_status(request_id: str, status: str):
    if request_id in requests_db:
        requests_db[request_id]["status"] = status
        requests_db[request_id]["updated_at"] = datetime.utcnow().isoformat()


@app.get("/")
async def index():
    return HTMLResponse((Path(__file__).parent / "index.html").read_text())


@app.post("/api/requests")
def create_request(body: RequestBody):
    req_id = str(uuid4())
    req = {
        "request_id": req_id,
        **body.model_dump(),
        "status": "CREATED",
        "created_at": datetime.utcnow().isoformat(),
        "updated_at": None,
        "result": None,
    }
    requests_db[req_id] = req

    # Register in shared pool so other agents can find this request
    if body.role == "seller":
        SELLERS.append({"sell_id": req_id, "user_id": body.user_id, "user_name": body.user_name,
                         "product_name": body.product_name, "city": body.city,
                         "lat": body.lat, "lon": body.lon,
                         "min_price": body.min_price, "max_price": body.max_price,
                         "with_in_kms": body.with_in_kms})
    else:
        BUYERS.append({"buyer_id": req_id, "user_id": body.user_id, "user_name": body.user_name,
                        "product_name": body.product_name, "city": body.city,
                        "lat": body.lat, "lon": body.lon,
                        "min_price": body.min_price, "max_price": body.max_price,
                        "with_in_kms": body.with_in_kms})
    return req


@app.get("/api/requests")
def list_requests(user_id: str):
    return [r for r in requests_db.values() if r["user_id"] == user_id]


# ── LLM helpers ────────────────────────────────────────────────────────────────

def _chat(system: str, conv: list) -> dict:
    history = "\n".join(
        f"[{m['role'].upper()}]: {m['content']} (offer: ${m.get('offer','N/A')})" for m in conv
    ) or "Start the negotiation with your opening offer."
    resp = llm.invoke([
        SystemMessage(content=system + '\nReply JSON only: {"offer":<n>,"message":"<t>","accept":<b>,"drop_off":<b>}'),
        HumanMessage(content=history),
    ])
    try:
        return json.loads(resp.content)
    except Exception:
        t = resp.content; s, e = t.find("{"), t.rfind("}")+1
        return json.loads(t[s:e]) if s!=-1 else {"offer":0,"message":t,"accept":False,"drop_off":False}


def _summarize(name, product, lo, hi, dialogs, side):
    other = "buyer" if side == "seller" else "seller"
    lines = "\n".join(
        f"  {other.title()} {i+1}: {d[f'{other}_name']} | {d['result']} | best=${d['best_offer']} | {d['rounds']} rounds"
        for i, d in enumerate(dialogs)
    )
    return llm.invoke([HumanMessage(content=(
        f"{side.title()} '{name}' {'selling' if side=='seller' else 'buying'} {product} (${lo}–${hi}).\n\n"
        f"Results:\n{lines}\n\n"
        f"In 2–3 sentences: which {other} is best and why?"
    ))]).content


# ── Agent runners (run in background thread) ───────────────────────────────────

def _negotiate_pair(my_system, their_system, my_role, their_role, send):
    """One negotiation session. Returns dialog dict."""
    conv, status = [], "ongoing"
    for _ in range(MAX_ROUNDS):
        a = _chat(my_system, conv)
        conv.append({"role": my_role, "content": a["message"], "offer": a.get("offer")})
        send({"type": "msg", "role": my_role, "content": a["message"], "offer": a.get("offer")})
        if a.get("drop_off"):
            status = "drop_off"; break
        b = _chat(their_system, conv)
        conv.append({"role": their_role, "content": b["message"], "offer": b.get("offer")})
        send({"type": "msg", "role": their_role, "content": b["message"], "offer": b.get("offer")})
        if b.get("accept") or a.get("accept"):
            status = "agreed"; break
    best_their = next((m["offer"] for m in reversed(conv) if m["role"] == their_role and m.get("offer")), None)
    rounds = sum(1 for m in conv if m["role"] == my_role)
    return {"status": status, "best_offer": best_their, "rounds": rounds}


def _run_agent(request_id: str, send, wait_human):
    req = requests_db[request_id]
    role = req["role"]

    # ── Search ──────────────────────────────────────────────────────────────
    _update_status(request_id, "SEARCHING")
    send({"type": "status_update", "status": "SEARCHING"})
    send({"type": "stream", "text": f"Searching for {'buyers' if role=='seller' else 'sellers'} within {req['with_in_kms']} km…"})

    pool_item = {**req, "sell_id" if role=="seller" else "buyer_id": request_id,
                 "product_name": req["product_name"]}
    matched = find_buyers_for_seller(pool_item) if role=="seller" else find_sellers_for_buyer(pool_item)

    if not matched:
        send({"type": "stream", "text": "No counterparts found in range."})
        _update_status(request_id, "NO_DEAL")
        send({"type": "status_update", "status": "NO_DEAL"})
        return

    # Stream each match found
    for m in matched:
        name = m.get("user_name","?"); city = m.get("city","?"); dist = m.get("distance_km","?")
        lo = m.get("min_price","?"); hi = m.get("max_price","?")
        send({"type": "match_found", "name": name, "city": city,
              "distance": dist, "product": m.get("product_name",""), "lo": lo, "hi": hi})

    # ── Negotiate ────────────────────────────────────────────────────────────
    _update_status(request_id, "NEGOTIATING")
    send({"type": "status_update", "status": "NEGOTIATING"})
    send({"type": "stream", "text": f"Starting negotiations with {len(matched)} agent(s)…"})

    my_sys_template = (
        f"You are a negotiation agent for {role} '{req['user_name']}' in {req['city']}.\n"
        f"Product: {req['product_name']}  |  Private range: ${req['min_price']}–${req['max_price']}\n"
        + ("Do NOT reveal your minimum. Open near max, concede slowly. Set accept=true if buyer meets floor. Set drop_off=true if hopeless."
           if role == "seller" else
           "Do NOT reveal your maximum. Open near min, concede slowly. Set accept=true if seller meets ceiling. Set drop_off=true if hopeless.")
    )

    dialogs = []
    for counterpart in matched:
        cname = counterpart.get("user_name","?")
        send({"type": "negotiation_start", "with": cname,
              "city": counterpart.get("city","?"), "distance": counterpart.get("distance_km","?")})

        their_sys = (
            f"You are a negotiation agent for {'buyer' if role=='seller' else 'seller'} '{cname}' in {counterpart.get('city','')}.\n"
            f"Product: {counterpart.get('product_name','')}  |  Private range: ${counterpart.get('min_price',0)}–${counterpart.get('max_price',0)}\n"
            + ("Do NOT reveal your maximum. Open near min, concede slowly. Set accept=true if seller meets ceiling."
               if role == "seller" else
               "Do NOT reveal your minimum. Open near max, concede slowly. Set accept=true if buyer meets floor.")
        )

        my_role_key    = "seller_robot" if role == "seller" else "buyer_robot"
        their_role_key = "buyer_robot"  if role == "seller" else "seller_robot"
        result = _negotiate_pair(my_sys_template, their_sys, my_role_key, their_role_key, send)

        key = "buyer_name" if role == "seller" else "seller_name"
        dialogs.append({key: cname, **result})
        send({"type": "negotiation_end", "with": cname, "result": result["status"], "best_offer": result["best_offer"]})

    # ── Summary + human decision ─────────────────────────────────────────────
    _update_status(request_id, "DEAL_FOUND")
    send({"type": "status_update", "status": "DEAL_FOUND"})
    summary = _summarize(req["user_name"], req["product_name"], req["min_price"], req["max_price"], dialogs, role)
    send({"type": "summary", "text": summary})

    other_key = "buyer_name" if role == "seller" else "seller_name"
    options = [f"{d[other_key]} — {d['status']} — best offer: ${d['best_offer']}" for d in dialogs]
    choice = wait_human(f"Which {'buyer' if role=='seller' else 'seller'} do you choose?", options)

    chosen = None
    raw = choice.strip()
    try:
        idx = int(raw) - 1
        chosen = dialogs[idx] if 0 <= idx < len(dialogs) else None
    except ValueError:
        chosen = next((d for d in dialogs if d[other_key].lower() == raw.lower()), None)

    if chosen:
        _update_status(request_id, "CLOSED")
        requests_db[request_id]["result"] = chosen
        send({"type": "status_update", "status": "CLOSED"})
        send({"type": "result", "status": "DEAL CLOSED",
              "with": chosen[other_key], "price": chosen["best_offer"]})
    else:
        _update_status(request_id, "NO_DEAL")
        send({"type": "status_update", "status": "NO_DEAL"})
        send({"type": "result", "status": "PASSED", "with": None, "price": None})

    # Clean up from pool
    if role == "seller":
        SELLERS[:] = [s for s in SELLERS if s["sell_id"] != request_id]
    else:
        BUYERS[:] = [b for b in BUYERS if b["buyer_id"] != request_id]


# ── WebSocket ──────────────────────────────────────────────────────────────────

@app.websocket("/ws/{request_id}")
async def ws_endpoint(websocket: WebSocket, request_id: str):
    await websocket.accept()
    if request_id not in requests_db:
        await websocket.send_json({"type": "error", "text": "Request not found."})
        await websocket.close()
        return

    loop = asyncio.get_event_loop()
    queue: asyncio.Queue = asyncio.Queue()
    interrupt_event = threading.Event()
    interrupt_resp: dict = {"value": None}
    ws_sessions[request_id] = {"interrupt_event": interrupt_event, "interrupt_resp": interrupt_resp}

    def send(msg: dict):
        asyncio.run_coroutine_threadsafe(queue.put(msg), loop).result()

    def wait_human(prompt: str, options: list) -> str:
        send({"type": "interrupt", "prompt": prompt, "options": options})
        interrupt_event.wait()
        interrupt_event.clear()
        return interrupt_resp["value"]

    threading.Thread(
        target=lambda: _agent_thread(request_id, send, wait_human, queue, loop),
        daemon=True
    ).start()

    async def sender():
        while True:
            msg = await queue.get()
            await websocket.send_json(msg)
            if msg["type"] in ("done", "error"):
                break

    async def receiver():
        while True:
            try:
                msg = await websocket.receive_json()
                if msg.get("type") == "resume":
                    interrupt_resp["value"] = msg["value"]
                    interrupt_event.set()
            except Exception:
                break

    try:
        await asyncio.gather(sender(), receiver())
    finally:
        ws_sessions.pop(request_id, None)


def _agent_thread(request_id, send, wait_human, queue, loop):
    try:
        _run_agent(request_id, send, wait_human)
    except Exception as ex:
        send({"type": "error", "text": str(ex)})
    finally:
        send({"type": "done"})
