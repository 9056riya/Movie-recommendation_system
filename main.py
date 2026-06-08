import asyncio
import os
import pickle
import certifi
from typing import Optional, List, Dict, Any, Tuple

import numpy as np
import pandas as pd
import httpx
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv


# =========================
# ENV
# =========================
load_dotenv()
TMDB_API_KEY = os.getenv("TMDB_API_KEY")

TMDB_BASE = "https://api.themoviedb.org/3"
TMDB_IMG_500 = "https://image.tmdb.org/t/p/w500"

if not TMDB_API_KEY:
    raise RuntimeError("TMDB_API_KEY missing. Put it in .env as TMDB_API_KEY=xxxx")


# =========================
# FASTAPI APP
# =========================
app = FastAPI(title="Movie Recommender API", version="5.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================
# PICKLE GLOBALS — TF-IDF
# =========================
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

DF_PATH          = os.path.join(MODELS_DIR, "df.pkl")
INDICES_PATH     = os.path.join(MODELS_DIR, "indices.pkl")
TFIDF_MATRIX_PATH= os.path.join(MODELS_DIR, "tfidf_matrix.pkl")
TFIDF_PATH       = os.path.join(MODELS_DIR, "tfidf.pkl")

df: Optional[pd.DataFrame] = None
indices_obj: Any            = None
tfidf_matrix: Any           = None
tfidf_obj: Any              = None
TITLE_TO_IDX: Optional[Dict[str, int]] = None

# =========================
# PICKLE GLOBALS — CF (SVD)
# =========================
CF_FACTORS_PATH  = os.path.join(MODELS_DIR, "cf_movie_factors.pkl")
CF_IDS_PATH      = os.path.join(MODELS_DIR, "cf_movie_ids.pkl")
CF_MOVIES_PATH   = os.path.join(MODELS_DIR, "cf_movies_df.pkl")

cf_movie_factors: Any                    = None
cf_movie_ids: Optional[List[int]]        = None
cf_movies_df: Optional[pd.DataFrame]     = None

# =========================
# SHARED HTTP CLIENT
# =========================
_http_client: Optional[httpx.AsyncClient] = None


# =========================
# PYDANTIC MODELS
# =========================
class TMDBMovieCard(BaseModel):
    tmdb_id: int
    title: str
    poster_url: Optional[str]    = None
    release_date: Optional[str]  = None
    vote_average: Optional[float]= None


class TMDBMovieDetails(BaseModel):
    tmdb_id: int
    title: str
    overview: Optional[str]      = None
    release_date: Optional[str]  = None
    poster_url: Optional[str]    = None
    backdrop_url: Optional[str]  = None
    genres: List[dict]           = []


class TFIDFRecItem(BaseModel):
    title: str
    score: float
    tmdb: Optional[TMDBMovieCard] = None


class CFRecItem(BaseModel):
    title: str
    score: float
    tmdb: Optional[TMDBMovieCard] = None


class SearchBundleResponse(BaseModel):
    query: str
    movie_details: TMDBMovieDetails
    tfidf_recommendations: List[TFIDFRecItem]
    cf_recommendations: List[CFRecItem]
    hybrid_recommendations: List[CFRecItem]
    genre_recommendations: List[TMDBMovieCard]


# =========================
# UTILS
# =========================
def _norm_title(t: str) -> str:
    return str(t).strip().lower()


def make_img_url(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    return f"{TMDB_IMG_500}{path}"


async def tmdb_get(path: str, params: Dict[str, Any]) -> Dict[str, Any]:
    global _http_client
    if _http_client is None:
        raise HTTPException(status_code=500, detail="HTTP client not initialized")

    q = dict(params)
    q["api_key"] = TMDB_API_KEY

    r = None
    for attempt in range(3):
        try:
            r = await _http_client.get(f"{TMDB_BASE}{path}", params=q)
            break
        except Exception as e:
            print(f"TMDB attempt {attempt + 1} failed: {repr(e)}")
            if attempt == 2:
                raise HTTPException(
                    status_code=502,
                    detail=f"TMDB request error after 3 attempts: {type(e).__name__} | {repr(e)}",
                )

    if r is None or r.status_code != 200:
        status = r.status_code if r is not None else "N/A"
        text   = r.text       if r is not None else ""
        raise HTTPException(status_code=502, detail=f"TMDB error {status}: {text}")

    return r.json()


async def tmdb_cards_from_results(results: List[dict], limit: int = 20) -> List[TMDBMovieCard]:
    out: List[TMDBMovieCard] = []
    for m in (results or [])[:limit]:
        out.append(TMDBMovieCard(
            tmdb_id=int(m["id"]),
            title=m.get("title") or m.get("name") or "",
            poster_url=make_img_url(m.get("poster_path")),
            release_date=m.get("release_date"),
            vote_average=m.get("vote_average"),
        ))
    return out


async def tmdb_movie_details(movie_id: int) -> TMDBMovieDetails:
    data = await tmdb_get(f"/movie/{movie_id}", {"language": "en-US"})
    return TMDBMovieDetails(
        tmdb_id=int(data["id"]),
        title=data.get("title") or "",
        overview=data.get("overview"),
        release_date=data.get("release_date"),
        poster_url=make_img_url(data.get("poster_path")),
        backdrop_url=make_img_url(data.get("backdrop_path")),
        genres=data.get("genres", []) or [],
    )


async def tmdb_search_movies(query: str, page: int = 1) -> Dict[str, Any]:
    return await tmdb_get(
        "/search/movie",
        {"query": query, "include_adult": "false", "language": "en-US", "page": page},
    )


async def tmdb_search_first(query: str) -> Optional[dict]:
    data    = await tmdb_search_movies(query=query, page=1)
    results = data.get("results", [])
    return results[0] if results else None


async def attach_tmdb_card_by_title(title: str) -> Optional[TMDBMovieCard]:
    try:
        m = await tmdb_search_first(title)
        if not m:
            return None
        return TMDBMovieCard(
            tmdb_id=int(m["id"]),
            title=m.get("title") or title,
            poster_url=make_img_url(m.get("poster_path")),
            release_date=m.get("release_date"),
            vote_average=m.get("vote_average"),
        )
    except Exception:
        return None


# =========================
# TF-IDF HELPERS
# =========================
def build_title_to_idx_map(indices: Any) -> Dict[str, int]:
    title_to_idx: Dict[str, int] = {}
    if isinstance(indices, dict):
        for k, v in indices.items():
            title_to_idx[_norm_title(k)] = int(v)
        return title_to_idx
    try:
        for k, v in indices.items():
            title_to_idx[_norm_title(k)] = int(v)
        return title_to_idx
    except Exception:
        raise RuntimeError("indices.pkl must be dict or pandas Series-like (with .items())")


def get_local_idx_by_title(title: str) -> int:
    global TITLE_TO_IDX
    if TITLE_TO_IDX is None:
        raise HTTPException(status_code=500, detail="TF-IDF index map not initialized")
    key = _norm_title(title)
    if key in TITLE_TO_IDX:
        return int(TITLE_TO_IDX[key])
    raise HTTPException(status_code=404, detail=f"Title not found in local dataset: '{title}'")


def tfidf_recommend_titles(query_title: str, top_n: int = 10) -> List[Tuple[str, float]]:
    global df, tfidf_matrix
    if df is None or tfidf_matrix is None:
        raise HTTPException(status_code=500, detail="TF-IDF resources not loaded")

    idx    = get_local_idx_by_title(query_title)
    qv     = tfidf_matrix[idx]
    scores = (tfidf_matrix @ qv.T).toarray().ravel()
    order  = np.argsort(-scores)

    out: List[Tuple[str, float]] = []
    for i in order:
        if int(i) == int(idx):
            continue
        try:
            title_i = str(df.iloc[int(i)]["title"])
        except Exception:
            continue
        out.append((title_i, float(scores[int(i)])))
        if len(out) >= top_n:
            break
    return out


# =========================
# CF (SVD) HELPERS
# =========================
def cf_recommend_titles(query_title: str, top_n: int = 10) -> List[Tuple[str, float]]:
    """
    SVD-based collaborative filtering recommendation.
    Finds movies with similar latent factor vectors.
    Returns [] if movie not found in CF dataset — never crashes.
    """
    global cf_movie_factors, cf_movie_ids, cf_movies_df

    if cf_movie_factors is None or cf_movie_ids is None or cf_movies_df is None:
        return []

    # Find movieId from title
    match = cf_movies_df[
        cf_movies_df["title"].str.lower().str.contains(query_title.lower(), regex=False)
    ]
    if match.empty:
        return []

    movie_id = match.iloc[0]["movieId"]
    if movie_id not in cf_movie_ids:
        return []

    idx          = cf_movie_ids.index(movie_id)
    query_vector = cf_movie_factors[idx]

    norms = np.linalg.norm(cf_movie_factors, axis=1)
    sims  = cf_movie_factors @ query_vector / (norms * np.linalg.norm(query_vector) + 1e-9)
    order = np.argsort(-sims)

    out: List[Tuple[str, float]] = []
    for i in order:
        if i == idx:
            continue
        mid       = cf_movie_ids[i]
        title_row = cf_movies_df[cf_movies_df["movieId"] == mid]
        if title_row.empty:
            continue
        # Strip year from MovieLens titles like "Toy Story (1995)"
        raw_title = title_row.iloc[0]["title"]
        clean     = raw_title.rsplit(" (", 1)[0] if " (" in raw_title else raw_title
        out.append((clean, round(float(sims[i]), 4)))
        if len(out) >= top_n:
            break
    return out


def hybrid_recommend_titles(
    query_title: str,
    top_n: int = 10,
    tfidf_weight: float = 0.4,
    cf_weight: float = 0.6,
) -> List[Tuple[str, float]]:
    """
    Hybrid = weighted combination of TF-IDF + CF scores.
    TF-IDF: content similarity (genres, overview, cast, director)
    CF:     user behaviour similarity (what users who liked this also liked)
    """
    # Get both recommendation lists (fetch more for better overlap)
    try:
        tfidf_recs = dict(tfidf_recommend_titles(query_title, top_n=top_n * 3))
    except Exception:
        tfidf_recs = {}

    cf_recs = dict(cf_recommend_titles(query_title, top_n=top_n * 3))

    if not tfidf_recs and not cf_recs:
        return []

    # Normalize scores to 0-1 range
    def normalize(d: dict) -> dict:
        if not d:
            return d
        max_v = max(d.values()) or 1
        return {k: v / max_v for k, v in d.items()}

    tfidf_norm = normalize(tfidf_recs)
    cf_norm    = normalize(cf_recs)

    # Combine all unique titles
    all_titles = set(tfidf_norm.keys()) | set(cf_norm.keys())

    hybrid: Dict[str, float] = {}
    for t in all_titles:
        hybrid[t] = (
            tfidf_weight * tfidf_norm.get(t, 0.0) +
            cf_weight    * cf_norm.get(t, 0.0)
        )

    sorted_recs = sorted(hybrid.items(), key=lambda x: -x[1])[:top_n]
    return [(t, round(s, 4)) for t, s in sorted_recs]


# =========================
# STARTUP / SHUTDOWN
# =========================
@app.on_event("startup")
async def startup():
    global _http_client, df, indices_obj, tfidf_matrix, tfidf_obj, TITLE_TO_IDX
    global cf_movie_factors, cf_movie_ids, cf_movies_df

    # Shared HTTP client
    _http_client = httpx.AsyncClient(
        timeout=12,
        verify=False,
        http2=False,
        limits=httpx.Limits(
            max_connections=50,
            max_keepalive_connections=20,
            keepalive_expiry=30,
        ),
    )

    # Load all pickles in a thread (non-blocking)
    loop = asyncio.get_event_loop()

    def _load():
        # TF-IDF
        with open(DF_PATH, "rb") as f:
            _df = pickle.load(f)
        with open(INDICES_PATH, "rb") as f:
            _indices = pickle.load(f)
        with open(TFIDF_MATRIX_PATH, "rb") as f:
            _matrix = pickle.load(f)
        with open(TFIDF_PATH, "rb") as f:
            _tfidf = pickle.load(f)

        # CF (SVD) — optional, won't crash if missing
        _cf_factors = _cf_ids = _cf_movies = None
        try:
            with open(CF_FACTORS_PATH, "rb") as f:
                _cf_factors = pickle.load(f)
            with open(CF_IDS_PATH, "rb") as f:
                _cf_ids = list(pickle.load(f))
            with open(CF_MOVIES_PATH, "rb") as f:
                _cf_movies = pickle.load(f)
        except FileNotFoundError:
            print(" CF model files not found — CF endpoints will return []")

        return _df, _indices, _matrix, _tfidf, _cf_factors, _cf_ids, _cf_movies

    df, indices_obj, tfidf_matrix, tfidf_obj, cf_movie_factors, cf_movie_ids, cf_movies_df = \
        await loop.run_in_executor(None, _load)

    TITLE_TO_IDX = build_title_to_idx_map(indices_obj)

    if df is None or "title" not in df.columns:
        raise RuntimeError("df.pkl must contain a DataFrame with a 'title' column")

    cf_status = f"{len(cf_movie_ids)} movies" if cf_movie_ids else "not loaded"
    print(f" Startup complete.")
    print(f"   TF-IDF: {len(df)} movies")
    print(f"   CF SVD: {cf_status}")


@app.on_event("shutdown")
async def shutdown():
    global _http_client
    if _http_client:
        await _http_client.aclose()
        print(" HTTP client closed.")


# =========================
# ROUTES
# =========================
@app.get("/health")
def health():
    return {
        "status": "ok",
        "tfidf_movies": len(df) if df is not None else 0,
        "cf_movies": len(cf_movie_ids) if cf_movie_ids else 0,
    }


# ---------- HOME FEED ----------
@app.get("/home", response_model=List[TMDBMovieCard])
async def home(
    category: str = Query("popular"),
    limit: int    = Query(24, ge=1, le=50),
):
    """
    Home feed. category: trending | popular | top_rated | upcoming | now_playing
    """
    try:
        if category == "trending":
            data = await tmdb_get("/trending/movie/day", {"language": "en-US"})
            return await tmdb_cards_from_results(data.get("results", []), limit=limit)

        if category not in {"popular", "top_rated", "upcoming", "now_playing"}:
            raise HTTPException(status_code=400, detail="Invalid category")

        data = await tmdb_get(f"/movie/{category}", {"language": "en-US", "page": 1})
        return await tmdb_cards_from_results(data.get("results", []), limit=limit)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Home route failed: {e}")


# ---------- TMDB KEYWORD SEARCH ----------
@app.get("/tmdb/search")
async def tmdb_search(
    query: str = Query(..., min_length=1),
    page: int  = Query(1, ge=1, le=10),
):
    return await tmdb_search_movies(query=query, page=page)


# ---------- MOVIE DETAILS ----------
@app.get("/movie/id/{tmdb_id}", response_model=TMDBMovieDetails)
async def movie_details_route(tmdb_id: int):
    return await tmdb_movie_details(tmdb_id)


# ---------- GENRE RECOMMENDATIONS ----------
@app.get("/recommend/genre", response_model=List[TMDBMovieCard])
async def recommend_genre(
    tmdb_id: int = Query(...),
    limit: int   = Query(18, ge=1, le=50),
):
    details = await tmdb_movie_details(tmdb_id)
    if not details.genres:
        return []

    genre_id = details.genres[0]["id"]
    discover = await tmdb_get(
        "/discover/movie",
        {"with_genres": genre_id, "language": "en-US", "sort_by": "popularity.desc", "page": 1},
    )
    cards = await tmdb_cards_from_results(discover.get("results", []), limit=limit)
    return [c for c in cards if c.tmdb_id != tmdb_id]


# ---------- TF-IDF ONLY ----------
@app.get("/recommend/tfidf")
async def recommend_tfidf(
    title: str = Query(..., min_length=1),
    top_n: int = Query(10, ge=1, le=50),
):
    recs = tfidf_recommend_titles(title, top_n=top_n)
    return [{"title": t, "score": s} for t, s in recs]


# ---------- CF (SVD) ONLY ----------
@app.get("/recommend/cf")
async def recommend_cf(
    title: str = Query(..., min_length=1),
    top_n: int = Query(10, ge=1, le=50),
):
    """
    Collaborative filtering recommendations using SVD.
    Based on user behaviour patterns from MovieLens 25M dataset.
    Returns [] if movie not in CF dataset.
    """
    recs = cf_recommend_titles(title, top_n=top_n)
    if not recs:
        return {"message": f"Movie '{title}' not found in CF dataset", "recommendations": []}
    return [{"title": t, "score": s} for t, s in recs]


# ---------- HYBRID ONLY ----------
@app.get("/recommend/hybrid")
async def recommend_hybrid(
    title: str        = Query(..., min_length=1),
    top_n: int        = Query(10, ge=1, le=50),
    tfidf_weight: float = Query(0.4, ge=0.0, le=1.0),
    cf_weight: float    = Query(0.6, ge=0.0, le=1.0),
):
    """
    Hybrid recommendations = TF-IDF (40%) + CF SVD (60%).
    Content similarity + user behaviour combined.
    Weights are adjustable.
    """
    recs = hybrid_recommend_titles(title, top_n=top_n, tfidf_weight=tfidf_weight, cf_weight=cf_weight)
    if not recs:
        raise HTTPException(status_code=404, detail=f"No recommendations found for: '{title}'")
    return [{"title": t, "score": s} for t, s in recs]


# ---------- FULL BUNDLE ----------
@app.get("/movie/search", response_model=SearchBundleResponse)
async def search_bundle(
    query: str  = Query(..., min_length=1),
    tfidf_top_n: int = Query(12, ge=1, le=30),
    cf_top_n: int    = Query(12, ge=1, le=30),
    genre_limit: int = Query(12, ge=1, le=30),
):
    """
    Full bundle for a selected movie:
      - TMDB movie details
      - TF-IDF recommendations + posters  (content based)
      - CF recommendations + posters      (collaborative filtering)
      - Hybrid recommendations + posters  (TF-IDF + CF combined)
      - Genre recommendations from TMDB
    All poster fetches run in parallel via asyncio.gather()
    """
    # Step 1: TMDB match
    best = await tmdb_search_first(query)
    if not best:
        raise HTTPException(status_code=404, detail=f"No TMDB movie found for: '{query}'")

    tmdb_id = int(best["id"])
    details = await tmdb_movie_details(tmdb_id)

    # Step 2: TF-IDF recs
    tfidf_recs: List[Tuple[str, float]] = []
    try:
        tfidf_recs = tfidf_recommend_titles(details.title, top_n=tfidf_top_n)
    except Exception:
        try:
            tfidf_recs = tfidf_recommend_titles(query, top_n=tfidf_top_n)
        except Exception:
            tfidf_recs = []

    # Step 3: CF recs
    cf_recs = cf_recommend_titles(details.title, top_n=cf_top_n)
    if not cf_recs:
        cf_recs = cf_recommend_titles(query, top_n=cf_top_n)

    # Step 4: Hybrid recs
    hybrid_recs = hybrid_recommend_titles(details.title, top_n=cf_top_n)
    if not hybrid_recs:
        hybrid_recs = hybrid_recommend_titles(query, top_n=cf_top_n)

    # Step 5: Fetch ALL posters in parallel
    all_titles = (
        [t for t, _ in tfidf_recs] +
        [t for t, _ in cf_recs] +
        [t for t, _ in hybrid_recs]
    )
    poster_tasks   = [attach_tmdb_card_by_title(t) for t in all_titles]
    poster_results = await asyncio.gather(*poster_tasks, return_exceptions=True)

    def safe_card(c: Any) -> Optional[TMDBMovieCard]:
        return None if isinstance(c, Exception) else c

    # Split poster results back
    n_tfidf  = len(tfidf_recs)
    n_cf     = len(cf_recs)

    tfidf_cards  = poster_results[:n_tfidf]
    cf_cards     = poster_results[n_tfidf:n_tfidf + n_cf]
    hybrid_cards = poster_results[n_tfidf + n_cf:]

    tfidf_items: List[TFIDFRecItem] = [
        TFIDFRecItem(title=t, score=s, tmdb=safe_card(c))
        for (t, s), c in zip(tfidf_recs, tfidf_cards)
    ]
    cf_items: List[CFRecItem] = [
        CFRecItem(title=t, score=s, tmdb=safe_card(c))
        for (t, s), c in zip(cf_recs, cf_cards)
    ]
    hybrid_items: List[CFRecItem] = [
        CFRecItem(title=t, score=s, tmdb=safe_card(c))
        for (t, s), c in zip(hybrid_recs, hybrid_cards)
    ]

    # Step 6: Genre recs
    genre_recs: List[TMDBMovieCard] = []
    if details.genres:
        genre_id = details.genres[0]["id"]
        discover = await tmdb_get(
            "/discover/movie",
            {"with_genres": genre_id, "language": "en-US", "sort_by": "popularity.desc", "page": 1},
        )
        cards      = await tmdb_cards_from_results(discover.get("results", []), limit=genre_limit)
        genre_recs = [c for c in cards if c.tmdb_id != details.tmdb_id]

    return SearchBundleResponse(
        query=query,
        movie_details=details,
        tfidf_recommendations=tfidf_items,
        cf_recommendations=cf_items,
        hybrid_recommendations=hybrid_items,
        genre_recommendations=genre_recs,
    )