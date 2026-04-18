"""
Interactive entry point.

Each user (seller or buyer) inputs their own details, which registers them in
the shared pool and initialises their personal agent. The agents then
communicate automatically via LLM. If they reach a deadlock the graph pauses
and the console prompts a human mediator before resuming.

Usage:
    OPENAI_API_KEY=<key> python -m sell_buy_agents.init_sellers_buyers
"""

from sell_buy_agents.buyer import BuyAgent, BuyRequest, Buyer
from sell_buy_agents.seller import SellAgent, SellRequest, Seller
from sell_buy_agents.seller_buyer_pool import BUYERS, SELLERS, diagnose_pool, print_diagnosis


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ask(prompt: str, default: str) -> str:
    value = input(f"  {prompt} [{default}]: ").strip()
    return value if value else default


def _ask_float(prompt: str, default: float) -> float:
    return float(_ask(prompt, str(default)))


def _divider(title: str = "") -> None:
    line = "=" * 54
    print(f"\n{line}")
    if title:
        print(f"  {title}")
        print(line)


# ---------------------------------------------------------------------------
# Seller init
# ---------------------------------------------------------------------------

def init_seller() -> tuple[Seller, SellAgent]:
    _divider("SELLER REGISTRATION")
    user_id      = _ask("Your user ID",         "U_S1")
    user_name    = _ask("Your name",            "Alice")
    product_name = _ask("Product to sell",      "iPhone 15")
    min_price    = _ask_float("Min price ($)",  750.0)
    max_price    = _ask_float("Max price ($)",  950.0)
    city         = _ask("Your city",            "San Francisco")
    lat          = _ask_float("Latitude",       37.7749)
    lon          = _ask_float("Longitude",      -122.4194)
    with_in_kms  = _ask_float("Search radius (km)", 10.0)

    request = SellRequest(
        user_id=user_id, user_name=user_name,
        product_name=product_name,
        min_price=min_price, max_price=max_price,
        city=city, lat=lat, lon=lon,
        with_in_kms=with_in_kms,
    )
    seller = Seller(request)

    # Register dynamically in the shared pool
    SELLERS.append({
        "sell_id": f"S_{user_id}", "user_id": user_id, "user_name": user_name,
        "product_name": product_name, "city": city,
        "lat": lat, "lon": lon,
        "min_price": min_price, "max_price": max_price,
        "with_in_kms": with_in_kms,
    })

    agent = SellAgent(agent_id=user_id)
    print(f"\n  ✓ Seller '{user_name}' registered — agent ready.")
    return seller, agent


# ---------------------------------------------------------------------------
# Buyer init
# ---------------------------------------------------------------------------

def init_buyer() -> tuple[Buyer, BuyAgent]:
    _divider("BUYER REGISTRATION")
    user_id      = _ask("Your user ID",              "U_B1")
    user_name    = _ask("Your name",                 "Bob")
    product_name = _ask("Product you want to buy",   "iPhone 15")
    min_price    = _ask_float("Lowest you'd pay ($)", 600.0)
    max_price    = _ask_float("Most you'd pay ($)",   860.0)
    city         = _ask("Your city",                 "Oakland")
    lat          = _ask_float("Latitude",            37.8044)
    lon          = _ask_float("Longitude",           -122.2712)
    with_in_kms  = _ask_float("Search radius (km)",  10.0)

    request = BuyRequest(
        user_id=user_id, user_name=user_name,
        product_name=product_name,
        min_price=min_price, max_price=max_price,
        city=city, lat=lat, lon=lon,
        with_in_kms=with_in_kms,
    )
    buyer = Buyer(request)

    # Register dynamically in the shared pool
    BUYERS.append({
        "buyer_id": f"B_{user_id}", "user_id": user_id, "user_name": user_name,
        "product_name": product_name, "city": city,
        "lat": lat, "lon": lon,
        "min_price": min_price, "max_price": max_price,
        "with_in_kms": with_in_kms,
    })

    agent = BuyAgent(agent_id=user_id)
    print(f"\n  ✓ Buyer '{user_name}' registered — agent ready.")
    return buyer, agent


# ---------------------------------------------------------------------------
# Run agents
# ---------------------------------------------------------------------------

def run_seller(seller: Seller, agent: SellAgent) -> None:
    sell_req_id = seller.create_sell_request()
    _divider(f"SELL AGENT RUNNING — {seller.user_name}")
    print(f"  Request ID : {sell_req_id}")
    print(f"  Product    : {seller.product}")
    print(f"  Price range: ${seller.min_price} – ${seller.max_price}")
    print(f"  Location   : {seller.city}  (radius {seller.with_in_kms} km)\n")

    result = agent.run(seller, sell_req_id)

    if result.get("action") == "accept":
        seller.close_sell_request(sell_req_id)

    _divider(f"SELL AGENT RESULT — {seller.user_name}")
    print(f"  Status : {result.get('status')}")
    print(f"  Market : {result.get('response')}")
    for d in result.get("dialogs", []):
        turns = len(d["conversation"])
        print(f"  Buyer '{d['buyer_name']}' → {d['result']}  ({turns} turn{'s' if turns != 1 else ''})")


def run_buyer(buyer: Buyer, agent: BuyAgent) -> None:
    buy_req_id = buyer.create_buy_request()
    _divider(f"BUY AGENT RUNNING — {buyer.user_name}")
    print(f"  Request ID : {buy_req_id}")
    print(f"  Product    : {buyer.product}")
    print(f"  Price range: ${buyer.min_price} – ${buyer.max_price}")
    print(f"  Location   : {buyer.city}  (radius {buyer.with_in_kms} km)\n")

    result = agent.run(buyer, buy_req_id)

    if result.get("action") == "accept":
        buyer.close_buy_request(buy_req_id)

    _divider(f"BUY AGENT RESULT — {buyer.user_name}")
    print(f"  Status : {result.get('status')}")
    print(f"  Market : {result.get('response')}")
    for d in result.get("dialogs", []):
        turns = len(d["conversation"])
        print(f"  Seller '{d['seller_name']}' → {d['result']}  ({turns} turn{'s' if turns != 1 else ''})")


# ---------------------------------------------------------------------------
# Diagnose
# ---------------------------------------------------------------------------

def _diagnose_interactive() -> None:
    _divider("DIAGNOSE POOL — search alive agents")
    print("  Leave any filter blank to skip it.\n")

    product   = input("  Product name (substring): ").strip() or None
    min_price = input("  Min price filter ($):     ").strip()
    max_price = input("  Max price filter ($):     ").strip()

    use_loc = input("  Filter by location? (y/n) [n]: ").strip().lower() == "y"
    lat = lon = radius_km = None
    if use_loc:
        lat       = float(input("  Latitude:         ").strip())
        lon       = float(input("  Longitude:        ").strip())
        radius_km = float(input("  Radius (km) [10]: ").strip() or "10")

    result = diagnose_pool(
        product_name=product,
        lat=lat, lon=lon, radius_km=radius_km,
        min_price=float(min_price) if min_price else None,
        max_price=float(max_price) if max_price else None,
    )
    print_diagnosis(result)


# ---------------------------------------------------------------------------
# Menu
# ---------------------------------------------------------------------------

def main() -> None:
    _divider("SELL-BUY AGENT PLATFORM")

    while True:
        print("  Options:")
        print("    1  Connect as seller  (register → find buyers → negotiate)")
        print("    2  Connect as buyer   (register → find sellers → negotiate)")
        print("    3  Diagnose pool      (search alive agents)")
        print("    4  Exit")
        choice = input("\n  Choice: ").strip()

        if choice == "1":
            seller, agent = init_seller()
            run_seller(seller, agent)

        elif choice == "2":
            buyer, agent = init_buyer()
            run_buyer(buyer, agent)

        elif choice == "3":
            _diagnose_interactive()

        elif choice == "4":
            print("\n  Goodbye.\n")
            break

        else:
            print("  Invalid choice.")


if __name__ == "__main__":
    main()
