
"""
Demo: 10 sellers × 10 buyers negotiate automatically — no human input.

For each seller the agent finds matched buyers within radius, negotiates
concurrently, then auto-picks the best outcome and prints a summary.

Run:
    python -m sell_buy_agents.demo
"""

import pathlib
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
load_dotenv(dotenv_path=pathlib.Path(__file__).parent / ".env")

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from sell_buy_agents.seller import _negotiate_with_one_buyer
from sell_buy_agents.seller_buyer_pool import (
    BUYERS, SELLERS, find_buyers_for_seller,
)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

# ---------------------------------------------------------------------------
# Seed the pool with 10 sellers and 10 buyers
# ---------------------------------------------------------------------------
# Five product categories, two sellers and two buyers each.
# Sellers and buyers in the same category are clustered within ~15 km.

SELLERS.clear()
BUYERS.clear()

SELLERS.extend([
    # iPhone 15 — SF / Oakland cluster
    {"sell_id": "S01", "user_id": "U01", "user_name": "Alice",
     "product_name": "iPhone 15", "city": "San Francisco",
     "lat": 37.7749, "lon": -122.4194, "min_price": 750, "max_price": 950, "with_in_kms": 15},
    {"sell_id": "S02", "user_id": "U02", "user_name": "Karen",
     "product_name": "iPhone 15", "city": "Oakland",
     "lat": 37.8044, "lon": -122.2712, "min_price": 700, "max_price": 900, "with_in_kms": 15},

    # MacBook Pro — San Jose / Santa Clara cluster
    {"sell_id": "S03", "user_id": "U03", "user_name": "Charlie",
     "product_name": "MacBook Pro", "city": "San Jose",
     "lat": 37.3382, "lon": -121.8863, "min_price": 1200, "max_price": 1600, "with_in_kms": 15},
    {"sell_id": "S04", "user_id": "U04", "user_name": "Mike",
     "product_name": "MacBook Pro", "city": "Santa Clara",
     "lat": 37.3541, "lon": -121.9552, "min_price": 1100, "max_price": 1500, "with_in_kms": 15},

    # iPad Air — Oakland / Berkeley cluster
    {"sell_id": "S05", "user_id": "U05", "user_name": "Eve",
     "product_name": "iPad Air", "city": "Oakland",
     "lat": 37.8044, "lon": -122.2712, "min_price": 400, "max_price": 600, "with_in_kms": 15},
    {"sell_id": "S06", "user_id": "U06", "user_name": "Nancy",
     "product_name": "iPad Air", "city": "Berkeley",
     "lat": 37.8716, "lon": -122.2727, "min_price": 350, "max_price": 550, "with_in_kms": 15},

    # AirPods Pro — Palo Alto / Sunnyvale cluster
    {"sell_id": "S07", "user_id": "U07", "user_name": "Grace",
     "product_name": "AirPods Pro", "city": "Sunnyvale",
     "lat": 37.3688, "lon": -122.0363, "min_price": 150, "max_price": 250, "with_in_kms": 15},
    {"sell_id": "S08", "user_id": "U08", "user_name": "Oscar",
     "product_name": "AirPods Pro", "city": "Palo Alto",
     "lat": 37.4419, "lon": -122.1430, "min_price": 120, "max_price": 220, "with_in_kms": 15},

    # Samsung TV — San Jose / Sunnyvale cluster
    {"sell_id": "S09", "user_id": "U09", "user_name": "Ivan",
     "product_name": "Samsung TV", "city": "San Jose",
     "lat": 37.3382, "lon": -121.8863, "min_price": 400, "max_price": 700, "with_in_kms": 15},
    {"sell_id": "S10", "user_id": "U10", "user_name": "Paula",
     "product_name": "Samsung TV", "city": "Santa Clara",
     "lat": 37.3541, "lon": -121.9552, "min_price": 350, "max_price": 650, "with_in_kms": 15},
])

BUYERS.extend([
    # iPhone 15 buyers
    {"buyer_id": "B01", "user_id": "V01", "user_name": "Bob",
     "product_name": "iPhone 15", "city": "Oakland",
     "lat": 37.8044, "lon": -122.2712, "min_price": 600, "max_price": 860, "with_in_kms": 15},
    {"buyer_id": "B02", "user_id": "V02", "user_name": "Liam",
     "product_name": "iPhone 15", "city": "San Francisco",
     "lat": 37.7749, "lon": -122.4194, "min_price": 650, "max_price": 850, "with_in_kms": 15},

    # MacBook Pro buyers
    {"buyer_id": "B03", "user_id": "V03", "user_name": "Diana",
     "product_name": "MacBook Pro", "city": "Santa Clara",
     "lat": 37.3541, "lon": -121.9552, "min_price": 1000, "max_price": 1500, "with_in_kms": 15},
    {"buyer_id": "B04", "user_id": "V04", "user_name": "Noah",
     "product_name": "MacBook Pro", "city": "Sunnyvale",
     "lat": 37.3688, "lon": -122.0363, "min_price": 1100, "max_price": 1450, "with_in_kms": 15},

    # iPad Air buyers
    {"buyer_id": "B05", "user_id": "V05", "user_name": "Frank",
     "product_name": "iPad Air", "city": "Berkeley",
     "lat": 37.8716, "lon": -122.2727, "min_price": 300, "max_price": 550, "with_in_kms": 15},
    {"buyer_id": "B06", "user_id": "V06", "user_name": "Olivia",
     "product_name": "iPad Air", "city": "Oakland",
     "lat": 37.8044, "lon": -122.2712, "min_price": 380, "max_price": 580, "with_in_kms": 15},

    # AirPods Pro buyers
    {"buyer_id": "B07", "user_id": "V07", "user_name": "Henry",
     "product_name": "AirPods Pro", "city": "Mountain View",
     "lat": 37.3861, "lon": -122.0839, "min_price": 100, "max_price": 200, "with_in_kms": 15},
    {"buyer_id": "B08", "user_id": "V08", "user_name": "Peter",
     "product_name": "AirPods Pro", "city": "Palo Alto",
     "lat": 37.4419, "lon": -122.1430, "min_price": 130, "max_price": 230, "with_in_kms": 15},

    # Samsung TV buyers
    {"buyer_id": "B09", "user_id": "V09", "user_name": "Judy",
     "product_name": "Samsung TV", "city": "Sunnyvale",
     "lat": 37.3688, "lon": -122.0363, "min_price": 300, "max_price": 600, "with_in_kms": 15},
    {"buyer_id": "B10", "user_id": "V10", "user_name": "Quinn",
     "product_name": "Samsung TV", "city": "Mountain View",
     "lat": 37.3861, "lon": -122.0839, "min_price": 320, "max_price": 620, "with_in_kms": 15},
])

# ---------------------------------------------------------------------------
# Per-seller demo runner
# ---------------------------------------------------------------------------

def _auto_pick(dialogs: list[dict]) -> dict | None:
    agreed = [d for d in dialogs if d["result"] == "agreed"]
    pool = agreed if agreed else dialogs
    return max(pool, key=lambda d: d["best_buyer_offer"] or 0, default=None)


def _summarize(seller: dict, dialogs: list[dict]) -> str:
    lines = "\n".join(
        f"  Buyer {i+1}: {d['buyer_name']} ({d['buyer_city']}) | "
        f"result={d['result']} | best offer=${d['best_buyer_offer']} | rounds={d['rounds']}"
        for i, d in enumerate(dialogs)
    )
    prompt = (
        f"Seller '{seller['user_name']}' sold {seller['product_name']} "
        f"(floor ${seller['min_price']}, ceiling ${seller['max_price']}).\n\n"
        f"Negotiation outcomes:\n{lines}\n\n"
        "In 2 sentences, say which buyer performed best and why."
    )
    return llm.invoke([HumanMessage(content=prompt)]).content


def run_seller(seller: dict) -> None:
    matched = find_buyers_for_seller(seller)
    if not matched:
        print(f"\n  [{seller['user_name']}] No buyers in range — skipping.\n")
        return

    W = 62
    print(f"\n{'═'*W}")
    print(f"  SELLER  {seller['user_name']} ({seller['city']})  |  {seller['product_name']}  |  ${seller['min_price']}–${seller['max_price']}")
    print(f"  Negotiating with {len(matched)} buyer(s) concurrently...")
    print(f"{'═'*W}")

    # Run all negotiations in parallel
    dialogs: list[dict | None] = [None] * len(matched)
    with ThreadPoolExecutor(max_workers=len(matched)) as ex:
        future_to_idx = {
            ex.submit(_negotiate_with_one_buyer, seller, buyer): idx
            for idx, buyer in enumerate(matched)
        }
        for future in as_completed(future_to_idx):
            dialogs[future_to_idx[future]] = future.result()

    # Print each conversation
    for d in dialogs:
        print(f"\n  ┌── Buyer: {d['buyer_name']} ({d['buyer_city']})  [{d['distance_km']} km]")
        for m in d["conversation"]:
            offer_str = f"  [${m['offer']}]" if m.get("offer") is not None else ""
            tag = m["role"].upper()
            print(f"  │  [{tag:6}] {m['content']}{offer_str}")
        print(f"  └── {d['result'].upper()}  |  best offer: ${d['best_buyer_offer']}  |  {d['rounds']} round(s)")

    # Auto-pick & summarise
    best = _auto_pick(dialogs)
    summary = _summarize(seller, dialogs)

    print(f"\n  ── AI SUMMARY ──")
    for line in summary.splitlines():
        print(f"  {line}")
    if best:
        print(f"\n  AUTO-SELECTED: {best['buyer_name']} @ ${best['best_buyer_offer']}")
    print(f"{'═'*W}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"\n{'█'*62}")
    print(f"  SELL-BUY DEMO  —  {len(SELLERS)} sellers  ×  {len(BUYERS)} buyers")
    print(f"  Agents negotiate automatically. No input needed.")
    print(f"{'█'*62}\n")

    for seller in SELLERS:
        run_seller(seller)


if __name__ == "__main__":
    main()
