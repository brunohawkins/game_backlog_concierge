from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple

import pandas as pd
import streamlit as st


APP_DIR = Path(__file__).parent
DATA_DIR = APP_DIR / "data"
STEAM_CSV = DATA_DIR / "steam.csv"
MEDIA_CSV = DATA_DIR / "steam_media_data.csv"


# -----------------------------
# Helpers
# -----------------------------
def split_semicolon(value: str) -> List[str]:
    if not isinstance(value, str) or not value.strip():
        return []
    return [x.strip() for x in value.split(";") if x.strip()]


def normalize_token(t: str) -> str:
    t = t.strip().lower()
    t = re.sub(r"[^a-z0-9+\- ]+", "", t)
    t = re.sub(r"\s+", " ", t)
    return t


def tokenize_free_text(s: str) -> Set[str]:
    if not s:
        return set()
    s = s.lower()
    tokens = re.findall(r"[a-z0-9]+(?:[+\-][a-z0-9]+)?", s)
    return {normalize_token(t) for t in tokens if len(t) >= 3}


def safe_ratio(pos: float, neg: float) -> float:
    denom = pos + neg
    if denom <= 0:
        return 0.0
    return pos / denom


def hours_label(hours: float) -> str:
    if hours <= 0:
        return "Unknown"
    if hours < 1:
        return f"{int(round(hours * 60))} mins"
    return f"{hours:.1f} hrs"


# Mood-to-keyword mapping using Steam-ish tag vocabulary
MOOD_KEYWORDS: Dict[str, List[str]] = {
    "Cozy": [
        "relaxing", "casual", "atmospheric", "cute", "family friendly",
        "exploration", "crafting", "farming", "walking simulator"
    ],
    "Competitive": [
        "competitive", "multiplayer", "pvp", "sports", "strategy",
        "tactical", "fps", "fighting"
    ],
    "Story": [
        "story rich", "rpg", "adventure", "narration", "visual novel",
        "choices matter", "great soundtrack"
    ],
    "Brainy": [
        "puzzle", "strategy", "tactical", "turn-based", "management",
        "simulation", "city builder"
    ],
    "Chaos": [
        "action", "fast-paced", "roguelike", "arcade", "shooter",
        "hack and slash", "bullet hell"
    ],
}


# -----------------------------
# Data loading / preparation (cached)
# -----------------------------
@st.cache_data(show_spinner=True)
def load_data() -> pd.DataFrame:
    if not STEAM_CSV.exists():
        raise FileNotFoundError(f"Missing {STEAM_CSV}. Put steam.csv in ./data/")
    if not MEDIA_CSV.exists():
        raise FileNotFoundError(f"Missing {MEDIA_CSV}. Put steam_media_data.csv in ./data/")

    steam = pd.read_csv(STEAM_CSV)
    media = pd.read_csv(MEDIA_CSV)

    # Join for cover images
    df = steam.merge(
        media[["steam_appid", "header_image"]],
        left_on="appid",
        right_on="steam_appid",
        how="left",
    )

    # Precompute lists and token sets
    df["platform_list"] = df["platforms"].apply(split_semicolon)
    df["genre_list"] = df["genres"].apply(split_semicolon)
    df["tag_list"] = df["steamspy_tags"].apply(split_semicolon)
    df["category_list"] = df["categories"].apply(split_semicolon)

    def to_token_set(row) -> Set[str]:
        toks = []
        toks.extend(row["genre_list"])
        toks.extend(row["tag_list"])
        return {normalize_token(t) for t in toks if t}

    df["token_set"] = df.apply(to_token_set, axis=1)

    # Multiplayer flags
    def is_multiplayer(categories: List[str]) -> bool:
        cats = {c.lower() for c in categories}
        multiplayer_markers = {
            "multi-player",
            "online multi-player",
            "local multi-player",
            "co-op",
            "online co-op",
            "local co-op",
            "shared/split screen",
        }
        return any(m in cats for m in multiplayer_markers)

    def is_singleplayer(categories: List[str]) -> bool:
        return "single-player" in {c.lower() for c in categories}

    df["is_multiplayer"] = df["category_list"].apply(is_multiplayer)
    df["is_singleplayer"] = df["category_list"].apply(is_singleplayer)

    # Ratings
    df["rating_ratio"] = df.apply(lambda r: safe_ratio(r["positive_ratings"], r["negative_ratings"]), axis=1)
    df["rating_volume"] = (df["positive_ratings"] + df["negative_ratings"]).clip(lower=0)

    # "Time to beat" proxy (hours) using typical Steam playtime (median preferred, fallback to average)
    df["typical_playtime_min"] = df["median_playtime"].fillna(0)
    df.loc[df["typical_playtime_min"] <= 0, "typical_playtime_min"] = df.loc[
        df["typical_playtime_min"] <= 0, "average_playtime"
    ].fillna(0)
    df["ttb_proxy_hours"] = (df["typical_playtime_min"].astype(float) / 60.0).clip(lower=0)

    # Clean minimal fields
    df["name"] = df["name"].fillna("Unknown")
    df["header_image"] = df["header_image"].fillna("")

    return df


def compute_score(
    row: pd.Series,
    desired_tokens: Set[str],
    mood_tokens: Set[str],
    target_hours: float | None,
    platform: str,
    multiplayer_pref: str,
) -> Tuple[float, List[str], List[str]]:
    """
    Returns: (score, reasons, matched_tokens_for_display)
    """
    score = 0.0
    reasons: List[str] = []

    tokens: Set[str] = row["token_set"]

    # Platform (hard match handled in filtering; small bonus here)
    if platform == "Any":
        score += 0.5
    else:
        score += 1.0
        reasons.append(f"Available on {platform}.")

    # Similarity: overlap with desired tokens (genres/tags + free text)
    overlap = tokens.intersection(desired_tokens)
    mood_overlap = tokens.intersection(mood_tokens)

    score += 2.5 * len(mood_overlap)
    score += 1.5 * len(overlap)

    # Length fit (time-to-beat proxy)
    # We do NOT assume you can only play short games in short sessions.
    # This is “overall length preference”, not “time available right now”.
    ttb_hours = float(row.get("ttb_proxy_hours", 0) or 0)

    if target_hours is not None and target_hours > 0:
        if ttb_hours > 0:
            ratio = ttb_hours / max(target_hours, 1e-6)
            # ratio=1 is perfect; farther away decays smoothly
            length_score = max(0.0, 3.0 - 2.0 * abs(math.log(ratio)))
            score += length_score
            reasons.append(f"Length aligns with ~{target_hours:.0f} hrs target.")
        else:
            # unknown length; neutral
            score += 0.2

    # Multiplayer preference (hard match handled in filtering; add explanation)
    if multiplayer_pref == "Solo":
        reasons.append("Single-player friendly.")
        score += 0.8
    elif multiplayer_pref == "With friends":
        reasons.append("Supports multiplayer/co-op.")
        score += 0.8
    else:
        score += 0.3

    # Quality signal: rating ratio + volume (gentle)
    ratio = float(row.get("rating_ratio", 0) or 0)
    vol = float(row.get("rating_volume", 0) or 0)
    score += 2.0 * ratio
    score += 0.25 * math.log10(1 + vol)

    matched_for_display = sorted(list((mood_overlap.union(overlap))))[:8]

    return score, reasons[:3], matched_for_display


# -----------------------------
# App
# -----------------------------
st.set_page_config(page_title="Game Backlog Concierge", page_icon="🎮", layout="wide")

st.title("Game Backlog Concierge")
st.caption("Prototype using Steam metadata: constraint filtering + similarity ranking. Built to ship fast.")

df = load_data()

# Build a reasonable genre list for the UI
all_genres = sorted(
    {g for sub in df["genre_list"].tolist() for g in sub if isinstance(g, str) and g.strip()}
)
genre_counts = df["genre_list"].explode().dropna().value_counts()
top_genres = [g for g in genre_counts.head(30).index.tolist() if g in all_genres]

left, right = st.columns([1, 2], gap="large")

with left:
    st.subheader("Your constraints")

    platform = st.selectbox("Platform", ["Any", "windows", "mac", "linux"], index=0)
    multiplayer_pref = st.selectbox("Multiplayer?", ["Either", "Solo", "With friends"], index=0)

    st.divider()
    st.subheader("Time to beat (preference)")

    ignore_length = st.checkbox("Ignore game length", value=False)
    if ignore_length:
        target_hours = None
    else:
        target_hours = st.number_input(
            "Target time to beat (hours)",
            min_value=1,
            max_value=300,
            value=20,
            step=5,
            help="Prototype uses Steam median/average playtime as a proxy for length. This is not official HowLongToBeat.",
        )

    st.divider()
    st.subheader("Your vibe")

    mood = st.selectbox("Mood", list(MOOD_KEYWORDS.keys()), index=0)
    preferred_genres = st.multiselect("Preferred genres (optional)", options=top_genres, default=[])
    vibe_text = st.text_input("Extra vibes (optional)", placeholder="e.g., roguelike, chill, story rich, fast-paced, puzzle")

    st.divider()
    min_rating = st.slider("Minimum rating ratio (optional)", 0.0, 1.0, 0.0, 0.05)
    max_price = st.slider("Max price (optional)", 0.0, float(df["price"].max()), float(min(20.0, df["price"].max())), 1.0)

    commit = st.button("Commit Mode: lock my pick", type="primary")


# -----------------------------
# Filtering
# -----------------------------
filtered = df.copy()

if platform != "Any":
    filtered = filtered[filtered["platform_list"].apply(lambda xs: platform in xs)]

if multiplayer_pref == "Solo":
    filtered = filtered[filtered["is_singleplayer"] == True]
elif multiplayer_pref == "With friends":
    filtered = filtered[filtered["is_multiplayer"] == True]

filtered = filtered[filtered["rating_ratio"] >= float(min_rating)]
filtered = filtered[filtered["price"] <= float(max_price)]

# Optional *soft* length filter to avoid absurd mismatches, without killing exploration
if target_hours is not None and target_hours > 0:
    # Allow unknown lengths; otherwise keep within ~10x either direction (very forgiving)
    def length_ok(h: float) -> bool:
        if h <= 0:
            return True
        return (h >= target_hours / 10.0) and (h <= target_hours * 10.0)

    filtered = filtered[filtered["ttb_proxy_hours"].apply(length_ok)]


# -----------------------------
# Similarity scoring
# -----------------------------
mood_tokens = {normalize_token(t) for t in MOOD_KEYWORDS[mood]}
genre_tokens = {normalize_token(g) for g in preferred_genres}
free_tokens = tokenize_free_text(vibe_text)

desired_tokens = mood_tokens.union(genre_tokens).union(free_tokens)

if desired_tokens:
    scored_rows = []
    for _, r in filtered.iterrows():
        score, reasons, matched_tokens = compute_score(
            r,
            desired_tokens=desired_tokens,
            mood_tokens=mood_tokens,
            target_hours=target_hours,
            platform=platform,
            multiplayer_pref=multiplayer_pref,
        )
        scored_rows.append((score, r, reasons, matched_tokens))

    scored_rows.sort(key=lambda x: x[0], reverse=True)
else:
    tmp = filtered.copy()
    tmp["fallback_score"] = 2.0 * tmp["rating_ratio"] + 0.25 * tmp["rating_volume"].apply(lambda v: math.log10(1 + float(v)))
    tmp = tmp.sort_values("fallback_score", ascending=False)
    scored_rows = [(float(r["fallback_score"]), r, ["High overall rating."], []) for _, r in tmp.iterrows()]

top3 = scored_rows[:3]

with right:
    if filtered.empty:
        st.warning("No games match your constraints. Try loosening platform/multiplayer/price filters.")
        st.stop()

    if commit and top3:
        best = top3[0]
        st.success(f"Tonight you're playing: **{best[1]['name']}**")
        if best[1].get("header_image"):
            st.image(best[1]["header_image"], width=420)

        st.write("**Why this fits:**")
        for rr in best[2]:
            st.write(f"- {rr}")

        if best[3]:
            st.write("**Matched tags:** " + ", ".join(best[3]))

        ttb = float(best[1].get("ttb_proxy_hours", 0) or 0)
        st.write(f"**Estimated time-to-beat (proxy):** {hours_label(ttb)}")
        st.write("---")

    st.subheader("Top picks")

    for idx, (score, r, reasons, matched) in enumerate(top3, start=1):
        with st.container(border=True):
            cols = st.columns([1, 2], gap="medium")
            with cols[0]:
                if r.get("header_image"):
                    st.image(r["header_image"], use_container_width=True)
                else:
                    st.caption("No cover image available.")

            with cols[1]:
                ttb = float(r.get("ttb_proxy_hours", 0) or 0)
                st.markdown(f"### {idx}. {r['name']}")
                st.caption(
                    f"Score: {score:.2f} · "
                    f"Est. time-to-beat (proxy): {hours_label(ttb)} · "
                    f"Rating ratio: {r.get('rating_ratio', 0):.2f} · "
                    f"Price: {r.get('price', 0):.2f}"
                )

                st.write("**Why it matches:**")
                for rr in reasons:
                    st.write(f"- {rr}")

                if matched:
                    st.write("**Matched tags:** " + ", ".join(matched))

                
    st.divider()
st.markdown(
    'Data: **"Steam Store Games (Clean dataset)"** by **Nik Davis** (Kaggle), licensed under **CC BY 4.0**. '
    '[Dataset link](https://www.kaggle.com/datasets/nikdavis/steam-store-games/data).  \n'
    'Not affiliated with Valve or Steam.'
)
