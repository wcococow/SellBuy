"""
POC seller/buyer pool — two plain Python lists, no database.

Seller fields : sell_id, user_id, user_name, product_name, city, lat, lon,
                min_price, max_price, with_in_kms
Buyer fields  : buyer_id, user_id, user_name, product_name, city, lat, lon,
                min_price, max_price, with_in_kms
"""
import math

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

SELLERS = [
    {
        "sell_id": "S001", "user_id": "U001", "user_name": "Alice",
        "product_name": "iPhone 15", "city": "San Francisco",
        "lat": 37.7749, "lon": -122.4194,
        "min_price": 750.0, "max_price": 950.0, "with_in_kms": 10.0,
    },
    {
        "sell_id": "S002", "user_id": "U003", "user_name": "Charlie",
        "product_name": "MacBook Pro", "city": "San Jose",
        "lat": 37.3382, "lon": -121.8863,
        "min_price": 1200.0, "max_price": 1600.0, "with_in_kms": 15.0,
    },
]

BUYERS = [
    {
        "buyer_id": "B001", "user_id": "U002", "user_name": "Bob",
        "product_name": "iPhone 15", "city": "Oakland",
        "lat": 37.8044, "lon": -122.2712,
        "min_price": 600.0, "max_price": 860.0, "with_in_kms": 10.0,
    },
    {
        "buyer_id": "B002", "user_id": "U004", "user_name": "Diana",
        "product_name": "iPhone 15", "city": "Berkeley",
        "lat": 37.8716, "lon": -122.2727,
        "min_price": 500.0, "max_price": 700.0, "with_in_kms": 10.0,  # gap with sellers → likely deadlock
    },
    {
        "buyer_id": "B003", "user_id": "U005", "user_name": "Eve",
        "product_name": "MacBook Pro", "city": "Santa Clara",
        "lat": 37.3541, "lon": -121.9552,
        "min_price": 1000.0, "max_price": 1500.0, "with_in_kms": 15.0,
    },
    {
        "buyer_id": "B004", "user_id": "U006", "user_name": "Frank",
        "product_name": "iPhone 15", "city": "Los Angeles",
        "lat": 34.0522, "lon": -118.2437,   # ~560 km — outside any seller radius
        "min_price": 700.0, "max_price": 900.0, "with_in_kms": 10.0,
    },
]

# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------

def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    a = (math.sin(math.radians(lat2 - lat1) / 2) ** 2
         + math.cos(phi1) * math.cos(phi2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2)
    return 2 * R * math.asin(math.sqrt(a))


def find_buyers_for_seller(seller: dict) -> list[dict]:
    """Return buyers wanting the same product within seller's with_in_kms radius."""
    matches = []
    for b in BUYERS:
        if b["product_name"].lower() != seller["product_name"].lower():
            continue
        dist = _haversine_km(seller["lat"], seller["lon"], b["lat"], b["lon"])
        if dist <= seller["with_in_kms"]:
            matches.append({**b, "distance_km": round(dist, 2)})
            print(f"  [Pool] Matched buyer '{b['user_name']}' ({b['city']}) — {dist:.2f} km")
    return matches


def find_sellers_for_buyer(buyer: dict) -> list[dict]:
    """Return sellers offering the same product within buyer's with_in_kms radius."""
    matches = []
    for s in SELLERS:
        if s["product_name"].lower() != buyer["product_name"].lower():
            continue
        dist = _haversine_km(buyer["lat"], buyer["lon"], s["lat"], s["lon"])
        if dist <= buyer["with_in_kms"]:
            matches.append({**s, "distance_km": round(dist, 2)})
            print(f"  [Pool] Matched seller '{s['user_name']}' ({s['city']}) — {dist:.2f} km")
    return matches


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def _price_overlap(a_min, a_max, b_min, b_max) -> bool:
    """True if two price ranges share any overlap."""
    return a_max >= b_min and b_max >= a_min


def diagnose_pool(
    product_name: str | None = None,
    lat: float | None = None,
    lon: float | None = None,
    radius_km: float | None = None,
    min_price: float | None = None,
    max_price: float | None = None,
) -> dict:
    """
    Return all alive sellers and buyers that satisfy every supplied filter.
    Filters are all optional and can be combined freely:
      product_name  — case-insensitive substring match
      lat/lon       — centre point; requires radius_km
      radius_km     — max distance from lat/lon
      min_price     — agent's max_price must be >= this
      max_price     — agent's min_price must be <= this
    Also annotates each matched seller with its live buyer count and vice-versa.
    """

    def _matches(agent: dict) -> tuple[bool, float | None]:
        if product_name and product_name.lower() not in agent["product_name"].lower():
            return False, None
        if min_price is not None and agent["max_price"] < min_price:
            return False, None
        if max_price is not None and agent["min_price"] > max_price:
            return False, None
        dist = None
        if lat is not None and lon is not None and radius_km is not None:
            dist = round(_haversine_km(lat, lon, agent["lat"], agent["lon"]), 2)
            if dist > radius_km:
                return False, None
        return True, dist

    matched_sellers = []
    for s in SELLERS:
        ok, dist = _matches(s)
        if not ok:
            continue
        live_buyers = find_buyers_for_seller(s)
        matched_sellers.append({
            **s,
            **({"distance_from_search_km": dist} if dist is not None else {}),
            "live_buyer_count": len(live_buyers),
            "live_buyers": [b["user_name"] for b in live_buyers],
            "price_gap_status": (
                "negotiable" if any(
                    _price_overlap(s["min_price"], s["max_price"], b["min_price"], b["max_price"])
                    for b in live_buyers
                ) else "gap" if live_buyers else "no_buyers"
            ),
        })

    matched_buyers = []
    for b in BUYERS:
        ok, dist = _matches(b)
        if not ok:
            continue
        live_sellers = find_sellers_for_buyer(b)
        matched_buyers.append({
            **b,
            **({"distance_from_search_km": dist} if dist is not None else {}),
            "live_seller_count": len(live_sellers),
            "live_sellers": [s["user_name"] for s in live_sellers],
            "price_gap_status": (
                "negotiable" if any(
                    _price_overlap(b["min_price"], b["max_price"], s["min_price"], s["max_price"])
                    for s in live_sellers
                ) else "gap" if live_sellers else "no_sellers"
            ),
        })

    return {"sellers": matched_sellers, "buyers": matched_buyers}


def print_diagnosis(result: dict) -> None:
    """Pretty-print the output of diagnose_pool()."""
    sellers = result["sellers"]
    buyers  = result["buyers"]

    W = 60
    print(f"\n{'='*W}")
    print(f"  POOL DIAGNOSIS  —  {len(sellers)} seller(s)  /  {len(buyers)} buyer(s) matched")
    print(f"{'='*W}")

    if sellers:
        print(f"\n  {'SELLERS':─<{W-2}}")
        for s in sellers:
            dist_str = f"  {s['distance_from_search_km']} km from search point" if "distance_from_search_km" in s else ""
            print(f"\n  [{s['sell_id']}] {s['user_name']}  —  {s['city']}{dist_str}")
            print(f"    Product    : {s['product_name']}")
            print(f"    Price range: ${s['min_price']} – ${s['max_price']}  (radius {s['with_in_kms']} km)")
            print(f"    Live buyers: {s['live_buyer_count']}  {s['live_buyers']}")
            print(f"    Price status: {s['price_gap_status'].upper()}")

    if buyers:
        print(f"\n  {'BUYERS':─<{W-2}}")
        for b in buyers:
            dist_str = f"  {b['distance_from_search_km']} km from search point" if "distance_from_search_km" in b else ""
            print(f"\n  [{b['buyer_id']}] {b['user_name']}  —  {b['city']}{dist_str}")
            print(f"    Product    : {b['product_name']}")
            print(f"    Price range: ${b['min_price']} – ${b['max_price']}  (radius {b['with_in_kms']} km)")
            print(f"    Live sellers: {b['live_seller_count']}  {b['live_sellers']}")
            print(f"    Price status: {b['price_gap_status'].upper()}")

    if not sellers and not buyers:
        print("\n  No agents matched your search criteria.")

    print(f"\n{'='*W}\n")
