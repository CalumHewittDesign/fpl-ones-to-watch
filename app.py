"""
FPL Ones to Watch
=================
A personal Fantasy Premier League recommender.

Tabs:
  1. Gameweek Picks  - top 5 players for the upcoming gameweek(s)
  2. Captains        - top 3 captain options for the upcoming gameweek only
  3. Differentials   - top 5 players under an ownership threshold
  4. Wildcard        - full 15-player squad optimised over the next few gameweeks
  5. Free Hit        - full 15-player squad optimised for the upcoming gameweek only

Data comes from the free, unofficial FPL API (no login needed).
Squad building uses PuLP (a linear-programming solver).
Injuries, suspensions and fitness are factored in - see the availability
section below and the CONFIG comments.

All the numbers you might want to tweak live in CONFIG just below.
"""

import datetime as dt
import re
from concurrent.futures import ThreadPoolExecutor

import base64
import os

import requests
import pandas as pd
import pulp
import streamlit as st
import streamlit.components.v1 as components

# ---------------------------------------------------------------------------
# CONFIG - tweak the app's behaviour here
# ---------------------------------------------------------------------------
CONFIG = {
    # How many future gameweeks to look at for Picks / Differentials / Wildcard
    "horizon_gws": 5,

    # Ownership ceiling (%) for the Differentials tab
    "differential_max_ownership": 15.0,

    # How many of a team's most recent finished matches count as "team form"
    "team_form_matches": 5,

    # How many finished gameweeks before the app fully trusts current-season
    # form (before that it blends in pre-season signals)
    "form_trust_gws": 3,

    # Bench players count for this fraction of a starter in squad optimisation
    "bench_weight": 0.15,

    # "Budget bench" price caps, in FPL API units (45 = 4.5m)
    "bench_price_caps": {"GKP": 45, "DEF": 45, "MID": 50, "FWD": 50},

    # Scoring weights once the season is underway (each set sums to 1.0)
    "weights_mature": {"form": 0.35, "fixture": 0.30, "team_form": 0.20, "ep": 0.15},
    "weights_captain_mature": {"form": 0.35, "fixture": 0.40, "team_form": 0.25},

    # Scoring weights for the early-season blend (no real form data yet)
    "weights_early": {"fixture": 0.40, "strength": 0.30, "price": 0.20, "ep": 0.10},
    "weights_captain_early": {"fixture": 0.45, "strength": 0.30, "price": 0.25},

    # Weight given to "importance to team" (share of team goals + assists)
    # inside the Differentials score
    "involvement_weight": 0.15,

    # ------------------- Availability (injuries/suspensions) ---------------
    # A player back from injury/suspension counts at this fraction for their
    # first week back (unless FPL's 25/50/75% flag says otherwise) ...
    "post_return_start": 0.75,
    # ... and recovers by this much each following gameweek, up to 100%
    "recovery_ramp_per_gw": 0.25,
    # If FPL says a player is out but gives no return date, later weeks of the
    # horizon are treated as a coin flip
    "unknown_absence_prob": 0.5,
    # Players below this average availability across the window are dropped
    "min_window_availability": 0.2,

    # ------------------- Fitness/rotation ramp -----------------------------
    # Judged on minutes over the last ramp_recent_gws gameweeks: full credit
    # at ramp_full_minutes, scaling down to ramp_floor for zero minutes
    "ramp_recent_gws": 3,
    "ramp_full_minutes": 180,
    "ramp_floor": 0.7,
    # ... and a player must have at least this many recent minutes to appear
    # in recommendations at all (season underway only)
    "gate_recent_minutes": 90,
    # How many top candidates get their match history fetched per refresh
    # (kept modest to stay polite to the FPL API)
    "shortlist_size": 200,

    # Fallback minutes gate for players outside the shortlist: at least this
    # share of possible season minutes
    "min_minutes_share": 0.5,
}

POSITIONS = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}
SQUAD_QUOTA = {"GKP": 2, "DEF": 5, "MID": 5, "FWD": 3}
XI_MIN = {"GKP": 1, "DEF": 3, "MID": 2, "FWD": 1}
XI_MAX = {"GKP": 1, "DEF": 5, "MID": 5, "FWD": 3}

API_BASE = "https://fantasy.premierleague.com/api/"
# Branding assets (the app falls back to plain text/emoji if these are
# missing from the repository, so nothing breaks without them)
LOGO_PATH = "logo.svg"
TOUCH_ICON_PATH = "static/apple-touch-icon.png"

# FPL sometimes rejects requests without a browser-like user agent
HEADERS = {"User-Agent": "Mozilla/5.0 (personal FPL recommender)"}


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------
@st.cache_data(ttl=1800, show_spinner="Fetching latest FPL data...")
def fetch_fpl_data():
    bootstrap = requests.get(API_BASE + "bootstrap-static/", headers=HEADERS, timeout=30)
    bootstrap.raise_for_status()
    fixtures = requests.get(API_BASE + "fixtures/", headers=HEADERS, timeout=30)
    fixtures.raise_for_status()
    return bootstrap.json(), fixtures.json()


@st.cache_data(ttl=1800, show_spinner="Checking recent minutes for top candidates...")
def fetch_recent_minutes(player_ids, since_round):
    """
    Minutes played after gameweek `since_round`, per player, from FPL's
    element-summary endpoint. Only called for a shortlist of top candidates.
    A player whose fetch fails maps to None (the caller falls back gracefully).
    """
    session = requests.Session()
    session.headers.update(HEADERS)

    def one(pid):
        try:
            r = session.get(f"{API_BASE}element-summary/{pid}/", timeout=15)
            r.raise_for_status()
            hist = r.json().get("history", [])
            return pid, sum(h.get("minutes", 0) for h in hist
                            if h.get("round", 0) > since_round)
        except Exception:
            return pid, None

    with ThreadPoolExecutor(max_workers=8) as ex:
        return dict(ex.map(one, player_ids))


# ---------------------------------------------------------------------------
# Availability (injuries, suspensions, fitness)
# ---------------------------------------------------------------------------
def play_probability(status, chance):
    """Chance of playing the NEXT gameweek, from FPL's own flags (0..1).
    status: a=available, d=doubtful, i=injured, s=suspended, u=unavailable."""
    if chance is not None:
        return max(0.0, min(chance / 100.0, 1.0))
    return {"a": 1.0, "d": 0.75}.get(status, 0.0)


def window_deadlines(events, start_gw, n_gws):
    """Deadline datetime for each gameweek in the window (extrapolates weekly
    if a deadline is missing, so the maths never falls over)."""
    by_id = {e["id"]: e.get("deadline_time") for e in events}
    out, prev = [], None
    for gw in range(start_gw, start_gw + n_gws):
        raw = by_id.get(gw)
        if raw:
            d = dt.datetime.fromisoformat(raw.replace("Z", "+00:00"))
        elif prev is not None:
            d = prev + dt.timedelta(days=7)
        else:
            d = dt.datetime.now(dt.timezone.utc) + dt.timedelta(days=7 * (gw - start_gw + 1))
        out.append(d)
        prev = d
    return out


def parse_return_date(news, anchor):
    """Pull a return date out of FPL's news text, e.g. 'Suspended until 30 Aug'
    or 'Knee injury - Expected back 12 Sep'. Returns None if there isn't one.
    `anchor` (the first deadline in the window) supplies the year."""
    if not news:
        return None
    m = re.search(r"(?:until|back)\s+(\d{1,2})\s+([A-Za-z]{3})", news)
    if not m:
        return None
    try:
        month = dt.datetime.strptime(m.group(2).title(), "%b").month
        d = dt.datetime(anchor.year, month, int(m.group(1)), tzinfo=dt.timezone.utc)
    except ValueError:
        return None
    if d < anchor - dt.timedelta(days=45):  # a date months in the past means next year
        d = d.replace(year=anchor.year + 1)
    return d


def availability_schedule(status, chance, return_date, deadlines):
    """
    Expected chance of playing (0..1) for each gameweek in the window.
      - status 'u' (left club / unavailable): 0 throughout
      - return date known from the news: 0 before it, then a ramp back to
        full fitness starting at post_return_start (or the FPL % if set)
      - FPL 25/50/75% flag with no date: that chance next week, recovering
        by recovery_ramp_per_gw each following week
      - out (injured/suspended) with no date: 0 next week, then a coin flip
      - fully fit: 1.0 throughout
    """
    ramp = CONFIG["recovery_ramp_per_gw"]
    n = len(deadlines)
    if status == "u":
        return [0.0] * n
    if return_date is not None:
        probs, p = [], None
        for d in deadlines:
            if d < return_date:
                probs.append(0.0)
            else:
                if p is None:
                    p = (chance / 100.0) if chance else CONFIG["post_return_start"]
                probs.append(min(1.0, p))
                p += ramp
        return probs
    p = play_probability(status, chance)
    if status in ("a", "d"):
        probs = []
        for _ in range(n):
            probs.append(min(1.0, p))
            p += ramp
        return probs
    return [p] + [CONFIG["unknown_absence_prob"]] * (n - 1)


def fit_label(status, chance, return_date, avail_next):
    """Short human-readable fitness note for the tables."""
    if status == "a" and chance in (None, 100):
        return "Fit"
    if avail_next == 0 and return_date is not None:
        return f"Out (back {return_date.strftime('%d %b')})"
    if avail_next == 0:
        return "Out"
    return f"{int(round(avail_next * 100))}%"


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------
def get_target_gw(events):
    """The gameweek we are recommending for (the next deadline)."""
    for e in events:
        if e.get("is_next"):
            return e["id"]
    for e in events:
        if not e.get("finished"):
            return e["id"]
    return events[-1]["id"]


def count_finished_gws(events):
    return sum(1 for e in events if e.get("finished"))


def compute_team_form(fixtures, n_matches):
    """Average goals scored/conceded per team over their last n finished matches."""
    finished = [
        f for f in fixtures
        if f.get("finished") and f.get("team_h_score") is not None and f.get("event")
    ]
    finished.sort(key=lambda f: (f["event"], f.get("kickoff_time") or ""))

    history = {}  # team_id -> list of (scored, conceded)
    for f in finished:
        history.setdefault(f["team_h"], []).append((f["team_h_score"], f["team_a_score"]))
        history.setdefault(f["team_a"], []).append((f["team_a_score"], f["team_h_score"]))

    form = {}
    for team_id, results in history.items():
        recent = results[-n_matches:]
        form[team_id] = {
            "scored": sum(r[0] for r in recent) / len(recent),
            "conceded": sum(r[1] for r in recent) / len(recent),
        }
    return form


def compute_fixture_ease(fixtures, start_gw, n_gws):
    """
    Per team: sum of (6 - difficulty) over every fixture in the window.
    A double gameweek adds two fixtures' worth of ease; a blank adds nothing,
    so doubles are rewarded and blanks are punished automatically.
    """
    end_gw = start_gw + n_gws - 1
    ease = {}
    opponents = {}
    for f in fixtures:
        gw = f.get("event")
        if gw is None or gw < start_gw or gw > end_gw:
            continue
        h, a = f["team_h"], f["team_a"]
        ease[h] = ease.get(h, 0) + (6 - f["team_h_difficulty"])
        ease[a] = ease.get(a, 0) + (6 - f["team_a_difficulty"])
        opponents.setdefault(h, []).append((a, "H"))
        opponents.setdefault(a, []).append((h, "A"))
    return ease, opponents


def normalise_into(df, col, out, pool_mask):
    """Scale df[col] to 0..1 using the pooled players' min/max."""
    sub = df.loc[pool_mask, col]
    rng = (sub.max() - sub.min()) or 1
    df[out] = ((df[col] - sub.min()) / rng).clip(0, 1)


def build_player_table(bootstrap, fixtures, horizon_gws):
    """One row per player with every raw and normalised metric the app needs."""
    events = bootstrap["events"]
    teams = {t["id"]: t for t in bootstrap["teams"]}
    target_gw = get_target_gw(events)
    finished_gws = count_finished_gws(events)

    team_form = compute_team_form(fixtures, CONFIG["team_form_matches"])
    ease_h, _ = compute_fixture_ease(fixtures, target_gw, horizon_gws)
    ease_1, opp_1 = compute_fixture_ease(fixtures, target_gw, 1)
    deadlines = window_deadlines(events, target_gw, horizon_gws)

    total_team_goals = {}
    for p in bootstrap["elements"]:
        total_team_goals[p["team"]] = (
            total_team_goals.get(p["team"], 0) + p["goals_scored"])

    rows = []
    for p in bootstrap["elements"]:
        team_id = p["team"]
        team = teams[team_id]
        pos = POSITIONS[p["element_type"]]
        opp_list = opp_1.get(team_id, [])
        next_fixture = ", ".join(
            f"{teams[o]['short_name']} ({venue})" for o, venue in opp_list
        ) or "BLANK"

        status = p.get("status", "a")
        chance = p.get("chance_of_playing_next_round")
        return_date = parse_return_date(p.get("news"), deadlines[0])
        schedule = availability_schedule(status, chance, return_date, deadlines)
        avail_next = schedule[0]
        avail_window = sum(schedule) / len(schedule)

        tf = team_form.get(team_id, {"scored": 0.0, "conceded": 0.0})
        # attackers care about goals scored; defenders/keepers about goals conceded
        if pos in ("MID", "FWD"):
            tf_raw = tf["scored"]
            strength = (team["strength_attack_home"] + team["strength_attack_away"]) / 2
        else:
            tf_raw = -tf["conceded"]  # fewer conceded = better, so negate
            strength = (team["strength_defence_home"] + team["strength_defence_away"]) / 2

        rows.append({
            "id": p["id"],
            "name": p["web_name"],
            "team_id": team_id,
            "team": team["short_name"],
            "pos": pos,
            "next_fixture": next_fixture,
            "fit": fit_label(status, chance, return_date, avail_next),
            "price": p["now_cost"] / 10.0,
            "price_raw": p["now_cost"],
            "form": float(p.get("form") or 0),
            "ep_next": float(p.get("ep_next") or 0),
            "ownership": float(p.get("selected_by_percent") or 0),
            "minutes": p.get("minutes", 0),
            "goals": p.get("goals_scored", 0),
            "assists": p.get("assists", 0),
            "status": status,
            "avail_next": avail_next,
            "avail_window": avail_window,
            "team_form_raw": tf_raw,
            "strength_raw": strength,
            "ease_h_raw": ease_h.get(team_id, 0),
            "ease_1_raw": ease_1.get(team_id, 0),
            "fixtures_this_gw": len(opp_list),
            "team_goals_total": total_team_goals.get(team_id, 0),
        })

    df = pd.DataFrame(rows)

    # Anyone effectively absent for most of the window drops out entirely
    df["available"] = df["avail_window"] > CONFIG["min_window_availability"]
    pool = df["available"]

    for col, out in [("form", "form_n"), ("ep_next", "ep_n"), ("price_raw", "price_n"),
                     ("ease_h_raw", "ease_h_n"), ("ease_1_raw", "ease_1_n")]:
        normalise_into(df, col, out, pool)

    # Team form and strength are normalised within attacker/defender groups,
    # because goals-conceded numbers only compare with other defences
    for raw, out in [("team_form_raw", "team_form_n"), ("strength_raw", "strength_n")]:
        df[out] = 0.5
        for att in (True, False):
            mask = df["pos"].isin(["MID", "FWD"]) if att else df["pos"].isin(["GKP", "DEF"])
            sub = df.loc[mask & pool, raw]
            if len(sub) and sub.max() != sub.min():
                df.loc[mask, out] = ((df.loc[mask, raw] - sub.min()) /
                                     (sub.max() - sub.min())).clip(0, 1)

    # Involvement: share of the team's goals a player has scored or assisted
    df["involvement"] = df.apply(
        lambda r: (r["goals"] + r["assists"]) / r["team_goals_total"]
        if r["team_goals_total"] > 0 else 0.0, axis=1)
    normalise_into(df, "involvement", "involvement_n", pool)

    # ------------------------------------------------------------------
    # Composite scores. alpha slides from 0 (pre-season) to 1 (trust real
    # current-season form) over the first few gameweeks.
    # ------------------------------------------------------------------
    alpha = min(finished_gws / CONFIG["form_trust_gws"], 1.0)
    wm, we = CONFIG["weights_mature"], CONFIG["weights_early"]
    cm, ce = CONFIG["weights_captain_mature"], CONFIG["weights_captain_early"]

    mature_h = (wm["form"] * df["form_n"] + wm["fixture"] * df["ease_h_n"]
                + wm["team_form"] * df["team_form_n"] + wm["ep"] * df["ep_n"])
    early_h = (we["fixture"] * df["ease_h_n"] + we["strength"] * df["strength_n"]
               + we["price"] * df["price_n"] + we["ep"] * df["ep_n"])
    df["score_horizon"] = (alpha * mature_h + (1 - alpha) * early_h) * df["avail_window"]

    mature_1 = (cm["form"] * df["form_n"] + cm["fixture"] * df["ease_1_n"]
                + cm["team_form"] * df["team_form_n"])
    early_1 = (ce["fixture"] * df["ease_1_n"] + ce["strength"] * df["strength_n"]
               + ce["price"] * df["price_n"])
    df["score_gw"] = (alpha * mature_1 + (1 - alpha) * early_1) * df["avail_next"]

    # ------------------------------------------------------------------
    # Fitness/rotation ramp, once the season is underway: fetch recent
    # minutes for the top candidates and scale scores accordingly
    # ------------------------------------------------------------------
    df["recent_minutes"] = pd.NA
    if finished_gws >= CONFIG["form_trust_gws"]:
        cand = df[df["available"]].copy()
        cand["best"] = cand[["score_horizon", "score_gw"]].max(axis=1)
        short_ids = tuple(sorted(
            cand.nlargest(CONFIG["shortlist_size"], "best")["id"].tolist()))
        since = finished_gws - CONFIG["ramp_recent_gws"]
        try:
            mins = fetch_recent_minutes(short_ids, since)
        except Exception:
            mins = {}
        known = {pid: m for pid, m in mins.items() if m is not None}
        df["recent_minutes"] = df["id"].map(known)

        floor, full = CONFIG["ramp_floor"], CONFIG["ramp_full_minutes"]
        ramp = df["recent_minutes"].map(
            lambda m: 1.0 if pd.isna(m) else floor + (1 - floor) * min(m / full, 1.0))
        df["score_horizon"] *= ramp
        df["score_gw"] *= ramp

        # Minutes gate: recent minutes where we have them, season share otherwise
        season_ok = df["minutes"] >= 90 * finished_gws * CONFIG["min_minutes_share"]
        recent_ok = df["recent_minutes"] >= CONFIG["gate_recent_minutes"]
        df["enough_minutes"] = recent_ok.where(df["recent_minutes"].notna(), season_ok)
    else:
        df["enough_minutes"] = True

    df["eligible"] = df["available"] & df["enough_minutes"].astype(bool)

    iw = CONFIG["involvement_weight"]
    df["score_diff"] = (1 - iw) * df["score_horizon"] + iw * df["involvement_n"]

    return df, target_gw, finished_gws, alpha


# ---------------------------------------------------------------------------
# Squad optimiser (Wildcard / Free Hit)
# ---------------------------------------------------------------------------
def solve_squad(df, budget_m, score_col, bench_mode):
    """
    Pick the best legal 15-man squad.
      - exactly 2 GKP / 5 DEF / 5 MID / 3 FWD
      - total price <= budget
      - max 3 players per club
      - a legal starting XI is chosen inside the squad; bench players count
        for CONFIG['bench_weight'] of a starter in the objective
      - bench_mode "Budget" additionally caps bench prices per position
    Returns (squad_df, status_message). squad_df has a 'starter' column.
    """
    players = df[df["available"]].reset_index(drop=True)
    budget_units = int(round(budget_m * 10))

    prob = pulp.LpProblem("fpl_squad", pulp.LpMaximize)
    x = {i: pulp.LpVariable(f"x_{i}", cat="Binary") for i in players.index}  # in squad
    y = {i: pulp.LpVariable(f"y_{i}", cat="Binary") for i in players.index}  # starts

    bw = CONFIG["bench_weight"]
    prob += pulp.lpSum(
        players.loc[i, score_col] * (y[i] + bw * (x[i] - y[i])) for i in players.index
    )

    prob += pulp.lpSum(players.loc[i, "price_raw"] * x[i] for i in players.index) <= budget_units
    prob += pulp.lpSum(y.values()) == 11

    for pos, quota in SQUAD_QUOTA.items():
        idx = players.index[players["pos"] == pos]
        prob += pulp.lpSum(x[i] for i in idx) == quota
        prob += pulp.lpSum(y[i] for i in idx) >= XI_MIN[pos]
        prob += pulp.lpSum(y[i] for i in idx) <= XI_MAX[pos]

    for team_id in players["team_id"].unique():
        idx = players.index[players["team_id"] == team_id]
        prob += pulp.lpSum(x[i] for i in idx) <= 3

    for i in players.index:
        prob += y[i] <= x[i]
        if bench_mode == "Budget":
            cap = CONFIG["bench_price_caps"][players.loc[i, "pos"]]
            # if a player is benched (x=1, y=0) their price must be under the cap
            prob += players.loc[i, "price_raw"] * (x[i] - y[i]) <= cap

    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    if pulp.LpStatus[prob.status] != "Optimal":
        return None, f"No legal squad found ({pulp.LpStatus[prob.status]}). Try a bigger budget."

    chosen = [i for i in players.index if x[i].value() == 1]
    squad = players.loc[chosen].copy()
    squad["starter"] = [y[i].value() == 1 for i in chosen]
    return squad, "ok"


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------
def show_table(df, score_col, extra_cols=None):
    cols = (["name", "team", "pos", "next_fixture", "fit", "price", "form", "ownership"]
            + (extra_cols or []) + [score_col])
    out = df[cols].rename(columns={
        "name": "Player", "team": "Team", "pos": "Pos", "next_fixture": "Next",
        "fit": "Fit", "price": "Price (£m)", "form": "Form", "ownership": "Owned %",
        "fixtures_this_gw": "Fixtures", score_col: "Score",
    })
    out["Score"] = out["Score"].round(3)
    st.dataframe(out, hide_index=True, width="stretch")


def show_squad(squad, score_col):
    pos_order = {"GKP": 0, "DEF": 1, "MID": 2, "FWD": 3}
    starters = squad[squad["starter"]].sort_values(
        by=["pos", score_col],
        key=lambda s: s.map(pos_order) if s.name == "pos" else s,
        ascending=[True, False])
    bench = squad[~squad["starter"]].sort_values(
        by=["pos", score_col],
        key=lambda s: s.map(pos_order) if s.name == "pos" else s,
        ascending=[True, False])

    formation = "-".join(str((starters["pos"] == p).sum()) for p in ["DEF", "MID", "FWD"])
    total = squad["price"].sum()
    c1, c2 = st.columns(2)
    c1.metric("Formation", formation)
    c2.metric("Squad cost", f"£{total:.1f}m")

    st.markdown("**Starting XI**")
    show_table(starters, score_col)
    st.markdown("**Bench**")
    show_table(bench, score_col)


# ---------------------------------------------------------------------------
# The app
# ---------------------------------------------------------------------------
def render_header():
    """Logo + title if logo.svg is in the repo, emoji title otherwise."""
    if os.path.exists(LOGO_PATH):
        b64 = base64.b64encode(open(LOGO_PATH, "rb").read()).decode()
        st.markdown(
            f"""<div style="display:flex;align-items:center;gap:14px;margin-bottom:0.5rem">
            <img src="data:image/svg+xml;base64,{b64}" style="height:56px" alt="logo">
            <h1 style="margin:0;padding:0">Ones to Watch</h1></div>""",
            unsafe_allow_html=True)
    else:
        st.title("⚽ FPL Ones to Watch")


def inject_touch_icon():
    """Tell iOS which icon to use for Add to Home Screen. Streamlit has no
    official way to set this, so a tiny script adds the tag to the page head.
    Harmless if it ever stops working (iOS just falls back to a screenshot)."""
    # components.html is proven to reach the parent page head; prefer it while
    # it exists, fall back to st.iframe if a future Streamlit removes it
    render_html = components.html if hasattr(components, "html") else st.iframe
    render_html(
        """<script>
        const doc = window.parent.document;
        if (!doc.querySelector("link[rel='apple-touch-icon']")) {
            const l = doc.createElement('link');
            l.rel = 'apple-touch-icon';
            l.sizes = '180x180';
            l.href = window.parent.location.origin + '/app/static/apple-touch-icon.png';
            doc.head.appendChild(l);
        }
        </script>""", height=0)


def main():
    icon = TOUCH_ICON_PATH if os.path.exists(TOUCH_ICON_PATH) else "⚽"
    st.set_page_config(page_title="Ones to Watch", page_icon=icon)
    render_header()
    inject_touch_icon()

    try:
        bootstrap, fixtures = fetch_fpl_data()
    except Exception as e:
        st.error(f"Could not reach the FPL API. Try again in a few minutes. ({e})")
        st.stop()

    df, target_gw, finished_gws, alpha = build_player_table(
        bootstrap, fixtures, CONFIG["horizon_gws"])
    pool = df[df["eligible"]]

    st.caption(
        f"Recommending for **Gameweek {target_gw}** · "
        f"{finished_gws} gameweek(s) finished · data refreshes every 30 min"
    )
    if alpha < 1:
        st.info(
            "Early-season mode: not enough matches played yet for reliable form "
            "data, so recommendations lean on fixtures, team strength and price. "
            f"Full form-based scoring kicks in after GW{CONFIG['form_trust_gws']}."
        )

    tab_picks, tab_caps, tab_diff, tab_wc, tab_fh = st.tabs(
        ["Picks", "Captains", "Differentials", "Wildcard", "Free Hit"])

    with tab_picks:
        st.subheader(f"Top 5 picks (GW{target_gw}–{target_gw + CONFIG['horizon_gws'] - 1})")
        show_table(pool.nlargest(5, "score_horizon"), "score_horizon")

    with tab_caps:
        st.subheader(f"Top 3 captains for GW{target_gw}")
        cap_pool = pool[pool["pos"].isin(["MID", "FWD"]) & (pool["fixtures_this_gw"] > 0)]
        show_table(cap_pool.nlargest(3, "score_gw"), "score_gw",
                   extra_cols=["fixtures_this_gw"])

    with tab_diff:
        st.subheader(f"Top 5 differentials (under {CONFIG['differential_max_ownership']:.0f}% owned)")
        diff_pool = pool[pool["ownership"] < CONFIG["differential_max_ownership"]]
        show_table(diff_pool.nlargest(5, "score_diff"), "score_diff")

    for tab, label, score_col, blurb in [
        (tab_wc, "Wildcard", "score_horizon",
         f"Optimised over the next {CONFIG['horizon_gws']} gameweeks. Injured or "
         "suspended players are discounted for the weeks they are expected to miss."),
        (tab_fh, "Free Hit", "score_gw",
         f"Optimised for GW{target_gw} only (double gameweeks handled automatically)."),
    ]:
        with tab:
            st.subheader(f"{label} squad builder")
            st.caption(blurb)
            budget = st.number_input(
                "Available funds (£m)", min_value=90.0, max_value=130.0,
                value=100.0, step=0.1, key=f"budget_{label}")
            bench_mode = st.radio(
                "Bench strategy", ["Budget", "Balanced"], key=f"bench_{label}",
                help="Budget: bench capped at 4.5 GK, 4.5 DEF, 5.0 MID, 5.0 FWD. "
                     "Balanced: no caps, but the starting XI is still prioritised.")
            if st.button(f"Build {label} squad", key=f"btn_{label}"):
                with st.spinner("Optimising squad..."):
                    squad, msg = solve_squad(df, budget, score_col, bench_mode)
                if squad is None:
                    st.error(msg)
                else:
                    show_squad(squad, score_col)


if __name__ == "__main__":
    main()
