"""
Each user initialises their own SellAgent or BuyAgent and runs it independently.

Usage:
    OPENAI_API_KEY=<key> python -m sell_buy_agents.main
"""

from sell_buy_agents.buyer import BuyAgent, BuyRequest, Buyer
from sell_buy_agents.seller import SellAgent, SellRequest, Seller


def main():
    # -----------------------------------------------------------------------
    # Seller user: Alice initialises her own SellAgent
    # -----------------------------------------------------------------------
    alice_request = SellRequest(
        user_id="U001", user_name="Alice",
        product_name="iPhone 15",
        min_price=750.0, max_price=950.0,
        city="San Francisco", lat=37.7749, lon=-122.4194,
        with_in_kms=10.0,
    )
    alice = Seller(alice_request)
    sell_req_id = alice.create_sell_request()

    print(f"\n[Alice] Created sell request: {sell_req_id}")
    print(f"[Alice] Status: {alice.sell_status}")

    alice_agent = SellAgent(agent_id=alice.user_id)
    sell_result = alice_agent.run(alice, sell_req_id)

    if sell_result.get("status") == "CLOSED" or sell_result.get("action") == "accept":
        alice.close_sell_request(sell_req_id)
    print(f"\n[Alice] Final status : {sell_result.get('status')}")
    print(f"[Alice] Market signal: {sell_result.get('response')}")
    for d in sell_result.get("dialogs", []):
        print(f"  → Buyer '{d['buyer_name']}': {d['result']}  ({len(d['conversation'])} turns)")

    # -----------------------------------------------------------------------
    # Buyer user: Bob initialises his own BuyAgent
    # -----------------------------------------------------------------------
    bob_request = BuyRequest(
        user_id="U002", user_name="Bob",
        product_name="iPhone 15",
        min_price=600.0, max_price=860.0,
        city="Oakland", lat=37.8044, lon=-122.2712,
        with_in_kms=10.0,
    )
    bob = Buyer(bob_request)
    buy_req_id = bob.create_buy_request()

    print(f"\n[Bob] Created buy request: {buy_req_id}")
    print(f"[Bob] Status: {bob.buy_status}")

    bob_agent = BuyAgent(agent_id=bob.user_id)
    buy_result = bob_agent.run(bob, buy_req_id)

    if buy_result.get("status") == "CLOSED" or buy_result.get("action") == "accept":
        bob.close_buy_request(buy_req_id)
    print(f"\n[Bob] Final status : {buy_result.get('status')}")
    print(f"[Bob] Market signal: {buy_result.get('response')}")
    for d in buy_result.get("dialogs", []):
        print(f"  → Seller '{d['seller_name']}': {d['result']}  ({len(d['conversation'])} turns)")


if __name__ == "__main__":
    main()
