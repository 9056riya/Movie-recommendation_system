import requests
import streamlit as st

# =============================
# CONFIG
# =============================
API_BASE = "http://127.0.0.1:8000"
TMDB_IMG = "https://image.tmdb.org/t/p/w500"

st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="wide")

# =============================
# STYLES
# =============================
st.markdown(
    """
<style>
.block-container { padding-top: 1rem; padding-bottom: 2rem; max-width: 1400px; }
.small-muted { color:#6b7280; font-size: 0.92rem; }
.movie-title { font-size: 0.9rem; line-height: 1.15rem; height: 2.3rem; overflow: hidden; }
.card { border: 1px solid rgba(0,0,0,0.08); border-radius: 16px; padding: 14px; background: rgba(255,255,255,0.7); }
.rec-header { font-size: 1.1rem; font-weight: 600; margin-bottom: 0.3rem; }
.badge-tfidf   { background:#dbeafe; color:#1d4ed8; padding:3px 10px; border-radius:999px; font-size:0.8rem; }
.badge-cf      { background:#dcfce7; color:#15803d; padding:3px 10px; border-radius:999px; font-size:0.8rem; }
.badge-hybrid  { background:#fef9c3; color:#854d0e; padding:3px 10px; border-radius:999px; font-size:0.8rem; }
.badge-genre   { background:#f3e8ff; color:#7e22ce; padding:3px 10px; border-radius:999px; font-size:0.8rem; }
</style>
""",
    unsafe_allow_html=True,
)

# =============================
# STATE + ROUTING
# =============================
if "view" not in st.session_state:
    st.session_state.view = "home"
if "selected_tmdb_id" not in st.session_state:
    st.session_state.selected_tmdb_id = None

qp_view = st.query_params.get("view")
qp_id   = st.query_params.get("id")
if qp_view in ("home", "details"):
    st.session_state.view = qp_view
if qp_id:
    try:
        st.session_state.selected_tmdb_id = int(qp_id)
        st.session_state.view = "details"
    except:
        pass


def goto_home():
    st.session_state.view = "home"
    st.query_params["view"] = "home"
    if "id" in st.query_params:
        del st.query_params["id"]
    st.rerun()


def goto_details(tmdb_id: int):
    st.session_state.view = "details"
    st.session_state.selected_tmdb_id = int(tmdb_id)
    st.query_params["view"] = "details"
    st.query_params["id"]   = str(int(tmdb_id))
    st.rerun()


# =============================
# API HELPERS
# =============================
@st.cache_data(ttl=30)
def api_get_json(path: str, params: dict | None = None):
    try:
        r = requests.get(f"{API_BASE}{path}", params=params, timeout=25)
        if r.status_code >= 400:
            return None, f"HTTP {r.status_code}: {r.text[:300]}"
        return r.json(), None
    except Exception as e:
        return None, f"Request failed: {e}"


def poster_grid(cards, cols=6, key_prefix="grid"):
    if not cards:
        st.info("No movies to show.")
        return

    rows = (len(cards) + cols - 1) // cols
    idx  = 0
    for r in range(rows):
        colset = st.columns(cols)
        for c in range(cols):
            if idx >= len(cards):
                break
            m       = cards[idx]
            idx    += 1
            tmdb_id = m.get("tmdb_id")
            title   = m.get("title", "Untitled")
            poster  = m.get("poster_url")

            with colset[c]:
                if poster:
                    st.image(poster, width="stretch")
                else:
                    st.write("🖼️ No poster")

                if st.button("Open", key=f"{key_prefix}_{r}_{c}_{idx}_{tmdb_id}"):
                    if tmdb_id:
                        goto_details(tmdb_id)

                st.markdown(
                    f"<div class='movie-title'>{title}</div>",
                    unsafe_allow_html=True,
                )


def to_cards_from_rec_items(items):
    """Converts tfidf_recommendations / cf_recommendations / hybrid_recommendations to cards."""
    cards = []
    for x in items or []:
        tmdb = x.get("tmdb") or {}
        if tmdb.get("tmdb_id"):
            cards.append({
                "tmdb_id":    tmdb["tmdb_id"],
                "title":      tmdb.get("title") or x.get("title") or "Untitled",
                "poster_url": tmdb.get("poster_url"),
            })
        else:
            # CF titles may not have tmdb attached — show title only
            cards.append({
                "tmdb_id":    None,
                "title":      x.get("title") or "Untitled",
                "poster_url": None,
            })
    return cards


def parse_tmdb_search_to_cards(data, keyword: str, limit: int = 24):
    keyword_l = keyword.strip().lower()

    if isinstance(data, dict) and "results" in data:
        raw = data.get("results") or []
        raw_items = []
        for m in raw:
            title   = (m.get("title") or "").strip()
            tmdb_id = m.get("id")
            poster_path = m.get("poster_path")
            if not title or not tmdb_id:
                continue
            raw_items.append({
                "tmdb_id":      int(tmdb_id),
                "title":        title,
                "poster_url":   f"{TMDB_IMG}{poster_path}" if poster_path else None,
                "release_date": m.get("release_date", ""),
            })
    elif isinstance(data, list):
        raw_items = []
        for m in data:
            tmdb_id = m.get("tmdb_id") or m.get("id")
            title   = (m.get("title") or "").strip()
            if not title or not tmdb_id:
                continue
            raw_items.append({
                "tmdb_id":      int(tmdb_id),
                "title":        title,
                "poster_url":   m.get("poster_url"),
                "release_date": m.get("release_date", ""),
            })
    else:
        return [], []

    matched    = [x for x in raw_items if keyword_l in x["title"].lower()]
    final_list = matched if matched else raw_items

    suggestions = []
    for x in final_list[:10]:
        year  = (x.get("release_date") or "")[:4]
        label = f"{x['title']} ({year})" if year else x["title"]
        suggestions.append((label, x["tmdb_id"]))

    cards = [
        {"tmdb_id": x["tmdb_id"], "title": x["title"], "poster_url": x["poster_url"]}
        for x in final_list[:limit]
    ]
    return suggestions, cards


# =============================
# SIDEBAR
# =============================
with st.sidebar:
    st.markdown("## 🎬 Menu")
    if st.button("🏠 Home"):
        goto_home()

    st.markdown("---")
    st.markdown("### 🏠 Home Feed")
    home_category = st.selectbox(
        "Category",
        ["trending", "popular", "top_rated", "now_playing", "upcoming"],
        index=0,
    )
    grid_cols = st.slider("Grid columns", 4, 8, 6)

    st.markdown("---")
    st.markdown("### ⚙️ Hybrid Weights")
    tfidf_w = st.slider("TF-IDF weight", 0.0, 1.0, 0.4, 0.1,
                        help="Content similarity weight")
    cf_w    = st.slider("CF weight",     0.0, 1.0, 0.6, 0.1,
                        help="User behaviour weight")
    st.caption(f"TF-IDF: {tfidf_w:.1f} | CF: {cf_w:.1f}")


# =============================
# HEADER
# =============================
st.title("🎬 Movie Recommender")
st.markdown(
    "<div class='small-muted'>Search → open a movie → see Content, Collaborative, Hybrid & Genre recommendations</div>",
    unsafe_allow_html=True,
)
st.divider()


# ==========================================================
# VIEW: HOME
# ==========================================================
if st.session_state.view == "home":
    typed = st.text_input(
        "Search by movie title", placeholder="Type: avenger, batman, inception..."
    )
    st.divider()

    if typed.strip():
        if len(typed.strip()) < 2:
            st.caption("Type at least 2 characters.")
        else:
            data, err = api_get_json("/tmdb/search", params={"query": typed.strip()})

            if err or data is None:
                st.error(f"Search failed: {err}")
            else:
                suggestions, cards = parse_tmdb_search_to_cards(data, typed.strip(), limit=24)

                if suggestions:
                    labels   = ["-- Select a movie --"] + [s[0] for s in suggestions]
                    selected = st.selectbox("Suggestions", labels, index=0)
                    if selected != "-- Select a movie --":
                        label_to_id = {s[0]: s[1] for s in suggestions}
                        goto_details(label_to_id[selected])
                else:
                    st.info("No suggestions found.")

                st.markdown("### Results")
                poster_grid(cards, cols=grid_cols, key_prefix="search_results")
        st.stop()

    st.markdown(f"### 🏠 {home_category.replace('_',' ').title()}")
    home_cards, err = api_get_json("/home", params={"category": home_category, "limit": 24})
    if err or not home_cards:
        st.error(f"Home feed failed: {err or 'Unknown error'}")
        st.stop()
    poster_grid(home_cards, cols=grid_cols, key_prefix="home_feed")


# ==========================================================
# VIEW: DETAILS
# ==========================================================
elif st.session_state.view == "details":
    tmdb_id = st.session_state.selected_tmdb_id
    if not tmdb_id:
        st.warning("No movie selected.")
        if st.button("← Back to Home"):
            goto_home()
        st.stop()

    a, b = st.columns([3, 1])
    with a:
        st.markdown("### 📄 Movie Details")
    with b:
        if st.button("← Back to Home"):
            goto_home()

    data, err = api_get_json(f"/movie/id/{tmdb_id}")
    if err or not data:
        st.error(f"Could not load details: {err or 'Unknown error'}")
        st.stop()

    # Poster + Info
    left, right = st.columns([1, 2.4], gap="large")
    with left:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        if data.get("poster_url"):
            st.image(data["poster_url"], width="stretch")
        else:
            st.write("🖼️ No poster")
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown(f"## {data.get('title','')}")
        release = data.get("release_date") or "-"
        genres  = ", ".join([g["name"] for g in data.get("genres", [])]) or "-"
        st.markdown(f"<div class='small-muted'>📅 Release: {release}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='small-muted'>🎭 Genres: {genres}</div>",   unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("### Overview")
        st.write(data.get("overview") or "No overview available.")
        st.markdown("</div>", unsafe_allow_html=True)

    if data.get("backdrop_url"):
        st.image(data["backdrop_url"], width="stretch")

    st.divider()

    # ======================================
    # RECOMMENDATIONS — 4 TABS
    # ======================================
    st.markdown("### 🍿 Recommendations")

    tab1, tab2, tab3, tab4 = st.tabs([
        "🔎 Content Based (TF-IDF)",
        "👥 Collaborative Filtering",
        "⚡ Hybrid",
        "🎭 By Genre",
    ])

    title = (data.get("title") or "").strip()

    # Fetch bundle once (has all 4 recommendation types)
    bundle, bundle_err = None, None
    if title:
        bundle, bundle_err = api_get_json(
            "/movie/search",
            params={
                "query":       title,
                "tfidf_top_n": 12,
                "cf_top_n":    12,
                "genre_limit": 12,
            },
        )

    # ---- TAB 1: TF-IDF ----
    with tab1:
        st.markdown(
            "<span class='badge-tfidf'>Content Based</span> "
            "Recommends movies similar in <b>plot, genre, cast & director</b>",
            unsafe_allow_html=True,
        )
        st.markdown("")
        if bundle and not bundle_err:
            cards = to_cards_from_rec_items(bundle.get("tfidf_recommendations"))
            poster_grid(cards, cols=grid_cols, key_prefix="tab_tfidf")
        else:
            st.warning(f"Could not load TF-IDF recommendations: {bundle_err}")

    # ---- TAB 2: Collaborative Filtering ----
    with tab2:
        st.markdown(
            "<span class='badge-cf'>Collaborative Filtering</span> "
            "Recommends based on <b>what users who liked this also watched</b> (SVD on MovieLens 25M)",
            unsafe_allow_html=True,
        )
        st.markdown("")
        if bundle and not bundle_err:
            cf_items = bundle.get("cf_recommendations") or []
            if cf_items:
                cards = to_cards_from_rec_items(cf_items)
                poster_grid(cards, cols=grid_cols, key_prefix="tab_cf")
            else:
                # Fallback: call /recommend/cf directly
                cf_data, cf_err = api_get_json("/recommend/cf", params={"title": title, "top_n": 12})
                if not cf_err and isinstance(cf_data, list) and cf_data:
                    # Build simple cards from title-only response
                    simple_cards = [
                        {"tmdb_id": None, "title": r["title"], "poster_url": None}
                        for r in cf_data
                    ]
                    st.caption("ℹ️ Posters unavailable for some titles — showing titles only.")
                    for c in simple_cards:
                        st.markdown(f"• {c['title']}")
                else:
                    st.info(
                        "This movie wasn't found in the MovieLens dataset. "
                        "CF works best for popular movies from 1995–2019."
                    )
        else:
            st.warning("Could not load CF recommendations.")

    # ---- TAB 3: Hybrid ----
    with tab3:
        st.markdown(
            f"<span class='badge-hybrid'>Hybrid</span> "
            f"TF-IDF <b>{int(tfidf_w*100)}%</b> + Collaborative Filtering <b>{int(cf_w*100)}%</b> "
            f"— adjust weights in the sidebar",
            unsafe_allow_html=True,
        )
        st.markdown("")
        if title:
            hybrid_data, hybrid_err = api_get_json(
                "/recommend/hybrid",
                params={
                    "title":        title,
                    "top_n":        12,
                    "tfidf_weight": tfidf_w,
                    "cf_weight":    cf_w,
                },
            )
            if not hybrid_err and isinstance(hybrid_data, list) and hybrid_data:
                # Fetch posters in parallel via bundle's hybrid section
                hybrid_items = bundle.get("hybrid_recommendations") if bundle else []
                if hybrid_items:
                    cards = to_cards_from_rec_items(hybrid_items)
                    poster_grid(cards, cols=grid_cols, key_prefix="tab_hybrid")
                else:
                    # No posters — show titles with scores
                    st.caption("ℹ️ Showing titles with scores (posters loading separately).")
                    for r in hybrid_data:
                        score_pct = int(r["score"] * 100)
                        st.markdown(f"• **{r['title']}** — match: {score_pct}%")
            else:
                st.info("No hybrid recommendations found for this movie.")
        else:
            st.warning("No title available.")

    # ---- TAB 4: Genre ----
    with tab4:
        st.markdown(
            "<span class='badge-genre'>By Genre</span> "
            "Popular movies in the <b>same genre</b> from TMDB",
            unsafe_allow_html=True,
        )
        st.markdown("")
        if bundle and not bundle_err:
            genre_cards = bundle.get("genre_recommendations") or []
            if genre_cards:
                poster_grid(genre_cards, cols=grid_cols, key_prefix="tab_genre")
            else:
                st.info("No genre recommendations available.")
        else:
            # Fallback: call /recommend/genre directly
            genre_data, genre_err = api_get_json(
                "/recommend/genre", params={"tmdb_id": tmdb_id, "limit": 18}
            )
            if not genre_err and genre_data:
                poster_grid(genre_data, cols=grid_cols, key_prefix="tab_genre_fallback")
            else:
                st.warning("No genre recommendations available right now.")