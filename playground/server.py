"""
FastAPI backend for the SellBuy personal portal - v2.

Start:
    cd /workspaces/SellBuy
    uvicorn playground.server:app --reload --port 8000
"""

import asyncio
import io
import json
import sys
import threading
from datetime import datetime
from pathlib import Path
from uuid import uuid4

# path must be set before sell_buy_agents is imported
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

import openai as _openai
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from sell_buy_agents.seller_buyer_pool import (
    BUYERS, SELLERS,
    find_buyers_for_seller, find_sellers_for_buyer,
)

llm       = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
MAX_ROUNDS = 5
app        = FastAPI()

# ── In-memory DB ──────────────────────────────────────────────────────────────
requests_db: dict[str, dict] = {}
sessions_db: dict[str, dict] = {}

# ── Seed pool ─────────────────────────────────────────────────────────────────
SELLERS.extend([
    {"sell_id": "seed-S1", "user_id": "seed", "user_name": "Alice",   "product_name": "iPhone 15",   "city": "San Francisco", "lat": 37.7749, "lon": -122.4194, "min_price": 750,  "max_price": 950,  "with_in_kms": 15},
    {"sell_id": "seed-S2", "user_id": "seed", "user_name": "Charlie", "product_name": "MacBook Pro",  "city": "San Jose",      "lat": 37.3382, "lon": -121.8863, "min_price": 1200, "max_price": 1600, "with_in_kms": 15},
    {"sell_id": "seed-S3", "user_id": "seed", "user_name": "Eve",     "product_name": "iPad Air",    "city": "Oakland",       "lat": 37.8044, "lon": -122.2712, "min_price": 400,  "max_price": 600,  "with_in_kms": 15},
    {"sell_id": "seed-S4", "user_id": "seed", "user_name": "Grace",   "product_name": "AirPods Pro", "city": "Sunnyvale",     "lat": 37.3688, "lon": -122.0363, "min_price": 150,  "max_price": 250,  "with_in_kms": 15},
    {"sell_id": "seed-S5", "user_id": "seed", "user_name": "Ivan",    "product_name": "Samsung TV",  "city": "San Jose",      "lat": 37.3382, "lon": -121.8863, "min_price": 400,  "max_price": 700,  "with_in_kms": 15},
])
BUYERS.extend([
    # SF / Bay Area buyers
    {"buyer_id": "seed-B1", "user_id": "seed", "user_name": "Bob",   "product_name": "iPhone 15",   "city": "Oakland",       "lat": 37.8044, "lon": -122.2712, "min_price": 600,  "max_price": 860,  "with_in_kms": 15},
    {"buyer_id": "seed-B2", "user_id": "seed", "user_name": "Diana", "product_name": "MacBook Pro",  "city": "Santa Clara",   "lat": 37.3541, "lon": -121.9552, "min_price": 1000, "max_price": 1500, "with_in_kms": 15},
    {"buyer_id": "seed-B3", "user_id": "seed", "user_name": "Frank", "product_name": "iPad Air",    "city": "Berkeley",      "lat": 37.8716, "lon": -122.2727, "min_price": 300,  "max_price": 550,  "with_in_kms": 15},
    {"buyer_id": "seed-B4", "user_id": "seed", "user_name": "Henry", "product_name": "AirPods Pro", "city": "Mountain View", "lat": 37.3861, "lon": -122.0839, "min_price": 100,  "max_price": 200,  "with_in_kms": 15},
    {"buyer_id": "seed-B5", "user_id": "seed", "user_name": "Judy",  "product_name": "Samsung TV",  "city": "Sunnyvale",     "lat": 37.3688, "lon": -122.0363, "min_price": 300,  "max_price": 600,  "with_in_kms": 15},
    # New York — Chair buyers (varied price ranges)
    {"buyer_id": "seed-B6",  "user_id": "seed", "user_name": "Emma",    "product_name": "Chair", "city": "Manhattan",  "lat": 40.7831, "lon": -73.9712, "min_price": 80,  "max_price": 200, "with_in_kms": 20},
    {"buyer_id": "seed-B7",  "user_id": "seed", "user_name": "Liam",    "product_name": "Chair", "city": "Brooklyn",   "lat": 40.6782, "lon": -73.9442, "min_price": 50,  "max_price": 150, "with_in_kms": 20},
    {"buyer_id": "seed-B8",  "user_id": "seed", "user_name": "Sophia",  "product_name": "Chair", "city": "Queens",     "lat": 40.7282, "lon": -73.7949, "min_price": 120, "max_price": 300, "with_in_kms": 20},
    {"buyer_id": "seed-B9",  "user_id": "seed", "user_name": "Noah",    "product_name": "Chair", "city": "Bronx",      "lat": 40.8448, "lon": -73.8648, "min_price": 60,  "max_price": 180, "with_in_kms": 20},
    {"buyer_id": "seed-B10", "user_id": "seed", "user_name": "Olivia",  "product_name": "Chair", "city": "Hoboken",    "lat": 40.7440, "lon": -74.0324, "min_price": 100, "max_price": 250, "with_in_kms": 20},
])

# New York — Chair sellers (so buyers can also find counterparts)
SELLERS.extend([
    {"sell_id": "seed-S6", "user_id": "seed", "user_name": "Marcus",  "product_name": "Chair", "city": "Manhattan",  "lat": 40.7580, "lon": -73.9855, "min_price": 90,  "max_price": 220, "with_in_kms": 20},
    {"sell_id": "seed-S7", "user_id": "seed", "user_name": "Priya",   "product_name": "Chair", "city": "Brooklyn",   "lat": 40.6501, "lon": -73.9496, "min_price": 60,  "max_price": 160, "with_in_kms": 20},
])


# ── Pydantic models ───────────────────────────────────────────────────────────

class RequestBody(BaseModel):
    user_id: str
    user_name: str
    role: str          # "seller" | "buyer"
    product_name: str
    min_price: float
    max_price: float
    city: str
    lat: float
    lon: float
    with_in_kms: float


class StartSessionsBody(BaseModel):
    request_id: str
    counterpart_ids: list[str]


class UpdateRequestBody(BaseModel):
    product_name: str
    min_price: float
    max_price: float
    city: str
    lat: float
    lon: float
    with_in_kms: float


class TTSBody(BaseModel):
    text: str
    content_type: str = "analysis"   # "analysis" | "summary" | "negotiation"
    role: str = "buyer"              # "buyer" | "seller"


# ── Match helpers ─────────────────────────────────────────────────────────────

def _find_matches(req: dict) -> list[dict]:
    role = req["role"]
    pool_item = {
        "product_name": req["product_name"],
        "lat": req["lat"], "lon": req["lon"],
        "with_in_kms": req["with_in_kms"],
    }
    if role == "seller":
        raw = find_buyers_for_seller(pool_item)
        return [{
            "counterpart_id": m["buyer_id"],
            "counterpart_role": "buyer",
            "name": m["user_name"],
            "city": m["city"],
            "distance_km": m.get("distance_km", 0),
            "product_name": m["product_name"],
            "min_price": m["min_price"],
            "max_price": m["max_price"],
            "with_in_kms": m.get("with_in_kms", 0),
            "_raw": m,
        } for m in raw]
    else:
        raw = find_sellers_for_buyer(pool_item)
        return [{
            "counterpart_id": m["sell_id"],
            "counterpart_role": "seller",
            "name": m["user_name"],
            "city": m["city"],
            "distance_km": m.get("distance_km", 0),
            "product_name": m["product_name"],
            "min_price": m["min_price"],
            "max_price": m["max_price"],
            "with_in_kms": m.get("with_in_kms", 0),
            "_raw": m,
        } for m in raw]


def _pub_req(req: dict) -> dict:
    return {**req, "matches": [{k: v for k, v in m.items() if k != "_raw"} for m in req.get("matches", [])]}


def _pub_sess(sess: dict) -> dict:
    return {k: v for k, v in sess.items() if not k.startswith("_")}


# ── REST endpoints ────────────────────────────────────────────────────────────

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
        "matches": [],
        "session_ids": [],
    }
    requests_db[req_id] = req

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

    req["matches"] = _find_matches(req)
    req["status"] = "MATCHED" if req["matches"] else "NO_MATCH"
    return _pub_req(req)


@app.get("/api/requests")
def list_requests(user_id: str):
    return [_pub_req(r) for r in requests_db.values() if r["user_id"] == user_id]


@app.post("/api/sessions")
def create_sessions(body: StartSessionsBody):
    req = requests_db.get(body.request_id)
    if not req:
        return {"error": "Request not found"}

    match_map = {m["counterpart_id"]: m for m in req.get("matches", [])}
    created = []
    for cp_id in body.counterpart_ids:
        cp = match_map.get(cp_id)
        if not cp:
            continue
        sess_id = str(uuid4())
        sess = {
            "session_id": sess_id,
            "request_id": body.request_id,
            "user_id": req["user_id"],
            "counterpart_id": cp_id,
            "counterpart_name": cp["name"],
            "counterpart_city": cp["city"],
            "counterpart_distance_km": cp["distance_km"],
            "counterpart_min_price": cp["min_price"],
            "counterpart_max_price": cp["max_price"],
            "status": "PENDING",
            "created_at": datetime.utcnow().isoformat(),
            "_raw": cp["_raw"],
        }
        sessions_db[sess_id] = sess
        req["session_ids"].append(sess_id)
        created.append(sess)

    req["status"] = "IN_PROGRESS"
    return [_pub_sess(s) for s in created]


@app.get("/api/sessions")
def list_sessions(request_id: str):
    return [_pub_sess(s) for s in sessions_db.values() if s["request_id"] == request_id]


@app.get("/api/sessions/{session_id}")
def get_session(session_id: str):
    sess = sessions_db.get(session_id)
    if not sess:
        return {"exists": False}
    return {"exists": True, **_pub_sess(sess)}


@app.put("/api/requests/{request_id}")
def update_request(request_id: str, body: UpdateRequestBody):
    req = requests_db.get(request_id)
    if not req:
        return {"error": "Not found"}

    # Remove stale pool entry and re-add with updated values
    role = req["role"]
    if role == "seller":
        SELLERS[:] = [s for s in SELLERS if s["sell_id"] != request_id]
        SELLERS.append({"sell_id": request_id, "user_id": req["user_id"],
                         "user_name": req["user_name"], **body.model_dump()})
    else:
        BUYERS[:] = [b for b in BUYERS if b["buyer_id"] != request_id]
        BUYERS.append({"buyer_id": request_id, "user_id": req["user_id"],
                        "user_name": req["user_name"], **body.model_dump()})

    req.update(body.model_dump())
    req["session_ids"] = []   # old sessions are invalid after edit
    req["matches"] = _find_matches(req)
    req["status"] = "MATCHED" if req["matches"] else "NO_MATCH"
    return _pub_req(req)


@app.post("/api/requests/{request_id}/analyze")
def analyze_deals(request_id: str):
    req = requests_db.get(request_id)
    if not req:
        return {"error": "Not found"}

    matches = req.get("matches", [])
    if not matches:
        return {"analysis": "No matches found to analyze."}

    role = req["role"]
    counterpart_role = "buyer" if role == "seller" else "seller"
    action = "selling" if role == "seller" else "buying"

    lines = []
    for i, m in enumerate(matches, 1):
        my_lo, my_hi = req["min_price"], req["max_price"]
        their_lo, their_hi = m["min_price"], m["max_price"]
        ovlp_lo, ovlp_hi = max(my_lo, their_lo), min(my_hi, their_hi)
        if ovlp_lo <= ovlp_hi:
            overlap_note = f"price overlap ${ovlp_lo}–${ovlp_hi} ✓"
        else:
            gap = ovlp_lo - ovlp_hi
            overlap_note = f"no price overlap (gap ~${gap:.0f}) ✗"
        lines.append(
            f"  {i}. {m['name']} in {m['city']} ({m['distance_km']} km away) — "
            f"range ${m['min_price']}–${m['max_price']} — {overlap_note}"
        )

    prompt = (
        f"You are a deal advisor. A user is {action} '{req['product_name']}' in {req['city']}.\n"
        f"Their price range: ${req['min_price']}–${req['max_price']}\n\n"
        f"Matched {counterpart_role}s:\n" + "\n".join(lines) + "\n\n"
        "Provide a concise analysis covering:\n"
        "1. Which deals are eligible (price ranges overlap) and why\n"
        "2. Your top recommendation and the reasoning (distance + price)\n"
        "3. Any deals to skip and why\n"
        "Be direct and specific. Use the names."
    )
    analysis = llm.invoke([HumanMessage(content=prompt)]).content
    return {"analysis": analysis, "count": len(matches)}


def _rewrite_for_voice(text: str, content_type: str, role: str) -> str:
    if role == "seller":
        goal = (
            "The listener is a SELLER. Their goal is to get the highest price possible "
            "while still closing the deal. Highlight which counterparts offer the best "
            "price opportunity, why those deals are worth pursuing, and gently warn about "
            "counterparts whose price range is too low or who may not be serious buyers."
        )
    else:
        goal = (
            "The listener is a BUYER. Their goal is to get the lowest reasonable price "
            "without sacrificing quality or reliability. Highlight which sellers offer "
            "the best value, flag any deals that look too expensive or risky, and guide "
            "them toward the smartest choice."
        )

    style = (
        "Speak like a calm, trusted personal advisor — warm, confident, and conversational. "
        "Do NOT use numbered lists or bullet points. Flow naturally from one thought to the next, "
        "using connecting phrases like 'what stands out here is…', 'the interesting thing about…', "
        "'if I were in your position…', 'worth keeping in mind…'. "
        "Keep it concise — under 120 words. Refer to people by name."
    )

    rewrite_prompt = (
        f"Rewrite the following deal analysis as a short spoken advisory.\n\n"
        f"Context: {goal}\n\nStyle: {style}\n\n"
        f"Original analysis:\n{text}\n\n"
        "Rewritten advisory (spoken, natural, no lists):"
    )
    return llm.invoke([HumanMessage(content=rewrite_prompt)]).content.strip()


@app.post("/api/tts")
async def text_to_speech(body: TTSBody):
    voice_map = {
        "analysis":    "nova",
        "summary":     "alloy",
        "negotiation": "shimmer",
    }
    voice = voice_map.get(body.content_type, "alloy")

    spoken_text = await asyncio.get_event_loop().run_in_executor(
        None, _rewrite_for_voice, body.text, body.content_type, body.role
    )

    client = _openai.OpenAI()
    audio_data = io.BytesIO()

    with client.audio.speech.with_streaming_response.create(
        model="tts-1",
        voice=voice,
        input=spoken_text[:4096],
    ) as response:
        for chunk in response.iter_bytes():
            audio_data.write(chunk)

    audio_data.seek(0)
    return StreamingResponse(audio_data, media_type="audio/mpeg",
                             headers={"Cache-Control": "no-store"})


# ── LLM helpers ───────────────────────────────────────────────────────────────

def _chat(system: str, conv: list) -> dict:
    history = "\n".join(
        f"[{m['role'].upper()}]: {m['content']} (offer: ${m.get('offer', 'N/A')})" for m in conv
    ) or "Start the negotiation with your opening offer."
    resp = llm.invoke([
        SystemMessage(content=system + '\nReply JSON only: {"offer":<n>,"message":"<t>","accept":<b>,"drop_off":<b>}'),
        HumanMessage(content=history),
    ])
    try:
        return json.loads(resp.content)
    except Exception:
        t = resp.content
        s, e = t.find("{"), t.rfind("}") + 1
        return json.loads(t[s:e]) if s != -1 else {"offer": 0, "message": t, "accept": False, "drop_off": False}


def _negotiate_pair(my_sys, their_sys, my_role, their_role, send):
    conv, status = [], "ongoing"
    for _ in range(MAX_ROUNDS):
        a = _chat(my_sys, conv)
        conv.append({"role": my_role, "content": a["message"], "offer": a.get("offer")})
        send({"type": "msg", "role": my_role, "content": a["message"], "offer": a.get("offer")})
        if a.get("drop_off"):
            status = "drop_off"
            break
        b = _chat(their_sys, conv)
        conv.append({"role": their_role, "content": b["message"], "offer": b.get("offer")})
        send({"type": "msg", "role": their_role, "content": b["message"], "offer": b.get("offer")})
        if b.get("accept") or a.get("accept"):
            status = "agreed"
            break
    best = next((m["offer"] for m in reversed(conv) if m["role"] == their_role and m.get("offer")), None)
    rounds = sum(1 for m in conv if m["role"] == my_role)
    return {"status": status, "best_offer": best, "rounds": rounds}


# ── Session runner ────────────────────────────────────────────────────────────

def _run_session(session_id: str, send, wait_human):
    sess = sessions_db[session_id]
    req  = requests_db[sess["request_id"]]
    role = req["role"]
    cp   = sess["_raw"]
    cname = cp.get("user_name", "?")

    sess["status"] = "NEGOTIATING"
    send({"type": "status_update", "status": "NEGOTIATING"})
    send({"type": "negotiation_start", "with": cname,
          "city": cp.get("city", "?"), "distance": cp.get("distance_km", "?")})

    my_sys = (
        f"You are a negotiation agent for {role} '{req['user_name']}' in {req['city']}.\n"
        f"Product: {req['product_name']} | Private range: ${req['min_price']}–${req['max_price']}\n"
        + ("Do NOT reveal your minimum. Open near max, concede slowly. "
           "Set accept=true if buyer meets floor. Set drop_off=true if hopeless."
           if role == "seller" else
           "Do NOT reveal your maximum. Open near min, concede slowly. "
           "Set accept=true if seller meets ceiling. Set drop_off=true if hopeless.")
    )
    their_sys = (
        f"You are a negotiation agent for {'buyer' if role == 'seller' else 'seller'} "
        f"'{cname}' in {cp.get('city', '')}.\n"
        f"Product: {cp.get('product_name', '')} | "
        f"Private range: ${cp.get('min_price', 0)}–${cp.get('max_price', 0)}\n"
        + ("Do NOT reveal your maximum. Open near min, concede slowly. "
           "Set accept=true if seller meets ceiling."
           if role == "seller" else
           "Do NOT reveal your minimum. Open near max, concede slowly. "
           "Set accept=true if buyer meets floor.")
    )

    my_role_key    = "seller_robot" if role == "seller" else "buyer_robot"
    their_role_key = "buyer_robot"  if role == "seller" else "seller_robot"
    result = _negotiate_pair(my_sys, their_sys, my_role_key, their_role_key, send)

    send({"type": "negotiation_end", "with": cname,
          "result": result["status"], "best_offer": result["best_offer"]})

    summary = llm.invoke([HumanMessage(content=(
        f"{role.title()} '{req['user_name']}' {'selling' if role == 'seller' else 'buying'} "
        f"{req['product_name']} (${req['min_price']}–${req['max_price']}).\n"
        f"Negotiation with {cname}: {result['status']} | "
        f"best offer: ${result['best_offer']} | {result['rounds']} rounds.\n"
        "In 1–2 sentences: should the user accept this deal?"
    ))]).content
    send({"type": "summary", "text": summary})

    sess["status"] = "AWAITING_DECISION"
    send({"type": "status_update", "status": "AWAITING_DECISION"})

    options = [
        f"Accept — close deal at ${result['best_offer']}",
        "Reject — pass on this deal",
    ]
    choice = wait_human(
        f"Negotiation ended: {result['status'].upper()} — best offer ${result['best_offer']}",
        options,
    )

    accepted = (choice or "").strip() == "1" or "accept" in (choice or "").lower()

    if accepted and result["status"] == "agreed":
        sess["status"] = "CLOSED"
        sess["result"] = {"with": cname, "price": result["best_offer"]}
        send({"type": "status_update", "status": "CLOSED"})
        send({"type": "result", "status": "DEAL CLOSED", "with": cname, "price": result["best_offer"]})
    else:
        sess["status"] = "NO_DEAL"
        send({"type": "status_update", "status": "NO_DEAL"})
        send({"type": "result", "status": "PASSED", "with": cname, "price": None})


# ── WebSocket (per session) ───────────────────────────────────────────────────

@app.websocket("/ws/{session_id}")
async def ws_endpoint(websocket: WebSocket, session_id: str):
    await websocket.accept()
    if session_id not in sessions_db:
        await websocket.send_json({"type": "error", "text": "Session not found."})
        await websocket.close()
        return

    loop             = asyncio.get_event_loop()
    queue: asyncio.Queue = asyncio.Queue()
    interrupt_event  = threading.Event()
    interrupt_resp   = {"value": None}

    def send(msg: dict):
        asyncio.run_coroutine_threadsafe(queue.put(msg), loop).result()

    def wait_human(prompt: str, options: list) -> str:
        send({"type": "interrupt", "prompt": prompt, "options": options})
        interrupt_event.wait()
        interrupt_event.clear()
        return interrupt_resp["value"]

    threading.Thread(
        target=lambda: _session_thread(session_id, send, wait_human),
        daemon=True,
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
    except Exception:
        pass


def _session_thread(session_id, send, wait_human):
    try:
        _run_session(session_id, send, wait_human)
    except Exception as ex:
        send({"type": "error", "text": str(ex)})
    finally:
        send({"type": "done"})
