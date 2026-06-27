"""
CommunalTable – Python/Flask backend with Remote Join (Socket.IO)
Run:  pip install flask flask-cors flask-socketio eventlet
      python server.py
"""

from flask import Flask, jsonify, send_from_directory, request, send_file
from flask_cors import CORS
from flask_socketio import SocketIO, join_room, leave_room, emit
import json, os, math, random, string, time, collections
app = Flask(__name__, static_folder="static", static_url_path="/static")
app.config['SECRET_KEY'] = 'communaltable-secret-key-change-me'
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')

BASE = os.path.dirname(__file__)

# ── load data once at startup ────────────────────────────────────────────────
with open(os.path.join(BASE, "public", "restaurants.json"), encoding="utf-8") as f:
    CATALOG = json.load(f)

try:
    with open(os.path.join(BASE, "public", "city-defaults.json"), encoding="utf-8") as f:
        CITY_DEFAULTS = json.load(f)
except Exception:
    CITY_DEFAULTS = {}

# ── in-memory room store ─────────────────────────────────────────────────────
# rooms[code] = {
#   "host_sid": str, "theme": str, "groupSize": int,
#   "city": str, "cityList": [..],
#   "members": [{cuisines, budgetMax, allergies}, ...],
#   "createdAt": float, "status": "waiting"|"done"
# }
ROOMS = {}

def gen_code(n=6):
    chars = string.ascii_uppercase + string.digits
    while True:
        code = ''.join(random.choices(chars, k=n))
        if code not in ROOMS:
            return code

# ── hardcoded top restaurant IDs per city (3 guaranteed picks) ───────────────
CITY_GUARANTEED = {}

def _build_guaranteed():
    by_city = {}
    for r in CATALOG:
        city = (r.get("c") or "").lower()
        if city:
            by_city.setdefault(city, []).append(r)
    for city, restaurants in by_city.items():
        top3 = sorted(restaurants, key=lambda x: -x.get("r", 0))[:3]
        CITY_GUARANTEED[city] = [r["i"] for r in top3]

_build_guaranteed()

# ── scoring constants ─────────────────────────────────────────────────────────
ALLERGY_MAP = {
    "dairy":    ["ice cream","desserts","sweets","bakery","kulfi"],
    "gluten":   ["pizzas","pastas","bakery","burgers","sandwich","rolls"],
    "nuts":     ["sweets","desserts","mughlai"],
    "seafood":  ["seafood","kerala","coastal","mangalorean","goan"],
    "egg":      ["bakery","desserts"],
    "shellfish":["seafood","chinese","thai"],
}

THEMES = {
    "family":   ["Thalis","South Indian","North Indian","Sweets","Bakery"],
    "pet":      ["Cafe","Continental","Beverages","Bakery","Healthy Food"],
    "romantic": ["Italian","Continental","Pastas","Desserts","Mediterranean"],
    "casual":   ["Pizzas","Fast Food","Chinese","Burgers","Snacks","Beverages"],
    "birthday": ["Pizzas","Desserts","Bakery","Ice Cream","Continental","Cafe"],
    "business": ["Continental","North Indian","Thalis","Healthy Food","Italian"],
    "latenight":["Fast Food","Biryani","Mughlai","Burgers","Chinese"],
    "brunch":   ["Cafe","Bakery","Continental","Beverages","Healthy Food","Desserts"],
    "solo":     ["Cafe","Bakery","Beverages","Desserts","Healthy Food"],
    "healthy":  ["Healthy Food","Salads","Cafe","Continental"],
}

CITY_COORDS = {
    "Mumbai":[19.076,72.8777],"Delhi":[28.6139,77.209],"New Delhi":[28.6139,77.209],
    "Bangalore":[12.9716,77.5946],"Hyderabad":[17.385,78.4867],"Chennai":[13.0827,80.2707],
    "Kolkata":[22.5726,88.3639],"Pune":[18.5204,73.8567],"Ahmedabad":[23.0225,72.5714],
    "Jaipur":[26.9124,75.7873],"Lucknow":[26.8467,80.9462],
}

def haversine(a, b):
    R = 6371
    dlat = math.radians(b[0]-a[0]); dlon = math.radians(b[1]-a[1])
    la1 = math.radians(a[0]); la2 = math.radians(b[0])
    x = math.sin(dlat/2)**2 + math.cos(la1)*math.cos(la2)*math.sin(dlon/2)**2
    return 2*R*math.asin(math.sqrt(x))

def real_dist_km(r, user_coords):
    """Real Haversine distance using per-restaurant lat/lng from dataset."""
    if not user_coords:
        return 5.0
    r_lat = r.get("lat")
    r_lng = r.get("lng")
    if r_lat is None or r_lng is None:
        return 5.0
    return haversine(user_coords, [r_lat, r_lng])

def _build_idf():
    """
    TF-IDF Feature Extraction — precomputed at startup. O(n x m).
    IDF(tag) = log((N+1)/(df+1)) + 1
    Rare tags (Awadhi, Chettinad) score higher than common ones (Beverages).
    """
    N = len(CATALOG)
    df = collections.Counter()
    for r in CATALOG:
        for tag in set(q.lower() for q in r.get("q", [])):
            df[tag] += 1
    return {tag: math.log((N+1)/(cnt+1))+1.0 for tag, cnt in df.items()}

CUISINE_IDF = _build_idf()

def cuisine_score(r, votes, boost, total):
    """TF-IDF weighted cuisine score. Rare specific cuisines rank higher than generic ones."""
    cuisines = [c.lower() for c in r.get("q", [])]
    sc = 0.0
    for c, cnt in votes.items():
        if c.lower() in cuisines:
            tf  = cnt / total
            idf = CUISINE_IDF.get(c.lower(), 1.0)
            sc += tf * idf * 0.5
    for c in boost:
        if c.lower() in cuisines:
            idf = CUISINE_IDF.get(c.lower(), 1.0)
            sc += 0.12 * min(idf / 2, 1.5)
    return max(0, min(1, sc))

def rating_score(r):
    rt = r.get("r", 3.5)
    return max(0, min(1, (rt - 2) / 3))

def budget_score(r, cap):
    p = r.get("p", 300)
    if p > cap: return 0
    return 1 - (p / cap) * 0.3

def dist_score(d):
    return max(0, 1 - d/25)

def composite(parts):
    """Weighted Linear Model — cuisine(TF-IDF) + rating + budget + distance(Haversine) + sentiment."""
    return round((parts["cuisine"]   * 0.33 +
                  parts["rating"]    * 0.25 +
                  parts["budget"]    * 0.18 +
                  parts["distance"]  * 0.14 +
                  parts["sentiment"] * 0.10) * 100)
def allergy_tokens(allergies):
    tokens = []
    for a in allergies:
        a_lower = a.lower()
        if a_lower in ALLERGY_MAP:
            tokens.extend(ALLERGY_MAP[a_lower])
        else:
            tokens.append(a_lower)
    return tokens

def compute_recommendations(members, theme_id, city_raw, city_list, filters=None):
    """Shared core recommendation engine — returns dict with picks, groupPrefs, filtersApplied."""
    filters = filters or {}
    min_rating  = float(filters.get("minRating", 3.5))
    max_dist_km = float(filters.get("maxDistanceKm", 999))
    sort_by     = filters.get("sortBy", "relevance")

    if not members:
        return {"picks": [], "fallback": True, "groupPrefs": {}, "filtersApplied": {}}

    cuisine_votes = {}
    budget_caps = []
    all_allergies = []
    all_cuisines_flat = []

    for m in members:
        for c in m.get("cuisines", []):
            cuisine_votes[c] = cuisine_votes.get(c, 0) + 1
            all_cuisines_flat.append(c)
        budget_caps.append(m.get("budgetMax", 600))
        all_allergies.extend(m.get("allergies", []))

    budget_cap = min(budget_caps) if budget_caps else 600
    total_voters = len(members)
    theme_boost = THEMES.get(theme_id, [])
    excluded = allergy_tokens(list(set(all_allergies)))

    city_norm = city_raw.lower() if city_raw else None
    city_list_norm = [c.lower() for c in city_list] if city_list else []
    user_lat = filters.get("userLat")
    user_lng = filters.get("userLng")
    if user_lat and user_lng:
        city_coords = [float(user_lat), float(user_lng)]
    else:
        city_coords = CITY_COORDS.get(city_raw) or (CITY_COORDS.get(city_list[0]) if city_list else None)

    def matches_place(r):
        c = (r.get("c") or "").lower()
        if city_norm: return city_norm in c
        if city_list_norm: return any(x in c for x in city_list_norm)
        return True

    def allergy_safe(r):
        blob = " ".join(r.get("q", [])).lower()
        return not any(tok in blob for tok in excluded)

    def score_r(r):
        d = real_dist_km(r, city_coords)
        # Sentiment boost from user feedback passed via filters
        fb_liked    = [x.lower() for x in filters.get("feedbackLiked", [])]
        fb_disliked = [x.lower() for x in filters.get("feedbackDisliked", [])]
        fb_sentiment = float(filters.get("feedbackSentiment", 0))
        tags = [q.lower() for q in r.get("q", [])]
        sent_score = 0.5
        if any(t in tags for t in fb_liked):    sent_score += 0.3
        if any(t in tags for t in fb_disliked): sent_score -= 0.3
        if fb_sentiment >  0.2 and r.get("r",0) >= 4.0: sent_score += 0.1
        if fb_sentiment < -0.2 and r.get("r",0) <  3.5: sent_score -= 0.1
        sent_score = max(0.0, min(1.0, sent_score))
        parts = {
            "cuisine":   cuisine_score(r, cuisine_votes, theme_boost, total_voters),
            "budget":    budget_score(r, budget_cap),
            "rating":    rating_score(r),
            "distance":  dist_score(d),
            "sentiment": sent_score,
        }
        return {"r": r, "score": composite(parts), "distanceKm": round(d,1), "parts": parts}
    pool = [r for r in CATALOG
            if r.get("p", 9999) <= budget_cap
            and matches_place(r)
            and allergy_safe(r)
            and r.get("r", 0) >= min_rating]

    scored = [score_r(r) for r in pool]

    if sort_by == "rating":
        scored.sort(key=lambda x: -x["r"]["r"])
    elif sort_by == "distance":
        scored.sort(key=lambda x: x["distanceKm"])
    else:
        scored.sort(key=lambda x: -x["score"])

    if max_dist_km < 900:
        scored = [s for s in scored if s["distanceKm"] <= max_dist_km]

    matched = [s for s in scored if s["parts"]["cuisine"] > 0][:3]
    fallback = len(matched) < 3

    if len(matched) < 3:
        seen = {s["r"]["i"] for s in matched}
        for s in scored:
            if s["r"]["i"] in seen: continue
            matched.append(s); seen.add(s["r"]["i"])
            if len(matched) == 3: break

    if len(matched) < 3:
        guaranteed_ids = (
            CITY_GUARANTEED.get(city_norm, []) or
            (CITY_GUARANTEED.get(city_list_norm[0]) if city_list_norm else []) or []
        )
        seen = {s["r"]["i"] for s in matched}
        for gid in guaranteed_ids:
            if gid in seen: continue
            r = next((x for x in CATALOG if x["i"] == gid), None)
            if r:
                matched.append(score_r(r)); seen.add(gid)
            if len(matched) == 3: break

    if len(matched) < 3:
        seen = {s["r"]["i"] for s in matched}
        for r in sorted(CATALOG, key=lambda x: -x.get("r",0)):
            if r["i"] in seen: continue
            matched.append(score_r(r)); seen.add(r["i"])
            if len(matched) == 3: break

    unique_cuisines = list(dict.fromkeys(all_cuisines_flat))
    group_prefs = {
        "cuisines":   unique_cuisines,
        "budgetMin":  min(budget_caps) if budget_caps else 0,
        "budgetMax":  max(budget_caps) if budget_caps else 600,
        "groupSize":  total_voters,
        "location":   city_raw or (city_list[0] if city_list else "Anywhere"),
    }

    filters_applied = {
        "distance": f"Within {int(max_dist_km)} km" if max_dist_km < 900 else "Any",
        "rating":   f"Above {min_rating} ★",
        "sortBy":   sort_by.capitalize(),
    }
    members.clear()  # Privacy: wipe member data from RAM after recommendations generated
    picks_out = [{"r": s["r"], "score": s["score"], "distanceKm": s["distanceKm"]} for s in matched]
    
    return {
        "picks": picks_out,
        "fallback": fallback,
        "groupPrefs": group_prefs,
        "filtersApplied": filters_applied,
    }

# ── API routes ────────────────────────────────────────────────────────────────

@app.route("/api/restaurants")
def api_restaurants():
    city = request.args.get("city", "").lower()
    limit = int(request.args.get("limit", 50))
    results = [r for r in CATALOG if not city or (r.get("c","").lower() == city)]
    return jsonify(results[:limit])

@app.route("/api/recommend", methods=["POST"])
def api_recommend():
    data = request.json or {}
    result = compute_recommendations(
        data.get("members", []),
        data.get("theme", "casual"),
        (data.get("city") or "").strip(),
        data.get("cityList") or [],
        data.get("filters", {}),
    )
    return jsonify(result)

@app.route("/api/city-defaults")
def api_city_defaults():
    return jsonify(CITY_DEFAULTS)

@app.route("/api/restaurant/<int:rid>")
def api_restaurant(rid):
    r = next((x for x in CATALOG if x.get("i") == rid), None)
    if not r:
        return jsonify({"error": "not found"}), 404
    return jsonify(r)

# ── REMOTE-JOIN REST endpoints ───────────────────────────────────────────────
@app.route("/api/room/<code>")
def api_room_info(code):
    """Guest hits this to discover the theme/city for a room."""
    code = code.upper()
    room = ROOMS.get(code)
    if not room:
        return jsonify({"error": "Room not found"}), 404
    return jsonify({
        "code": code,
        "theme": room["theme"],
        "groupSize": room["groupSize"],
        "city": room.get("city", ""),
        "cityList": room.get("cityList", []),
        "submitted": len(room["members"]),
        "status": room["status"],
    })

# ── SPA route for join page (same index.html — JS reads ?join=CODE) ──────────
@app.route("/join/<code>")
def join_page(code):
    return send_file(os.path.join(BASE, "index.html"))

# ── SOCKET.IO events ─────────────────────────────────────────────────────────
@socketio.on("connect")
def on_connect():
    emit("connected", {"sid": request.sid})

@socketio.on("create_room")
def on_create_room(data):
    """Host creates a room. data = {theme, groupSize, city, cityList}"""
    code = gen_code()
    ROOMS[code] = {
        "host_sid": request.sid,
        "theme": data.get("theme", "casual"),
        "groupSize": max(2, min(20, int(data.get("groupSize", 3)))),
        "city": (data.get("city") or "").strip(),
        "cityList": data.get("cityList") or [],
        "members": [],
        "createdAt": time.time(),
        "status": "waiting",
    }
    join_room(code)
    emit("room_created", {
        "code": code,
        "theme": ROOMS[code]["theme"],
        "groupSize": ROOMS[code]["groupSize"],
        "city": ROOMS[code]["city"],
    })

@socketio.on("join_room_as_guest")
def on_join_room(data):
    """Guest joins. data = {code}"""
    code = (data.get("code") or "").upper()
    room = ROOMS.get(code)
    if not room:
        emit("join_error", {"error": "Room not found. Check the code."})
        return
    if room["status"] == "done":
        emit("join_error", {"error": "This room has already finished."})
        return
    if len(room["members"]) >= room["groupSize"]:
        emit("join_error", {"error": "This room is full."})
        return
    join_room(code)
    emit("joined_room", {
        "code": code,
        "theme": room["theme"],
        "groupSize": room["groupSize"],
        "city": room.get("city", ""),
        "submitted": len(room["members"]),
    })
    # notify host of guest count
    socketio.emit("guest_joined", {
        "submitted": len(room["members"]),
        "total": room["groupSize"],
    }, room=code)

@socketio.on("submit_guest_prefs")
def on_submit_prefs(data):
    """Guest submits prefs. data = {code, cuisines, budgetMax, allergies}"""
    code = (data.get("code") or "").upper()
    room = ROOMS.get(code)
    if not room:
        emit("submit_error", {"error": "Room not found."})
        return
    if room["status"] == "done":
        emit("submit_error", {"error": "Room already finished."})
        return
    if len(room["members"]) >= room["groupSize"]:
        emit("submit_error", {"error": "Room is already full."})
        return

    member = {
        "cuisines":  data.get("cuisines", []),
        "budgetMax": int(data.get("budgetMax", 600)),
        "allergies": data.get("allergies", []),
    }
    room["members"].append(member)
    submitted = len(room["members"])
    total = room["groupSize"]

    emit("submit_ok", {"submitted": submitted, "total": total})

    socketio.emit("progress_update", {
        "submitted": submitted,
        "total": total,
    }, room=code)

    # all members in → compute and broadcast
    if submitted >= total:
        room["status"] = "done"
        result = compute_recommendations(
            room["members"],
            room["theme"],
            room.get("city", ""),
            room.get("cityList", []),
        )
        socketio.emit("room_result", result, room=code)

@socketio.on("close_room")
def on_close_room(data):
    code = (data.get("code") or "").upper()
    if code in ROOMS and ROOMS[code]["host_sid"] == request.sid:
        socketio.emit("room_closed", {"code": code}, room=code)
        del ROOMS[code]

# ── static / SPA ─────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return send_file(os.path.join(BASE, "index.html"))

@app.route("/public/<path:filename>")
def public_files(filename):
    return send_from_directory(os.path.join(BASE, "public"), filename)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    socketio.run(app, host='0.0.0.0', port=port, debug=False)