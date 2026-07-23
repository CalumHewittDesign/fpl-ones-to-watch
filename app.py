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

All the numbers you might want to tweak live in CONFIG just below.
"""

import requests
import pandas as pd
import pulp
import streamlit as st

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
    # form (before that it blends in pre-season signals - see notes in guide)
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

    # Minutes gate once form is trusted: a player must have played at least
    # this share of possible minutes to be recommended
    "min_minutes_share": 0.5,
}

POSITIONS = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}
SQUAD_QUOTA = {"GKP": 2, "DEF": 5, "MID": 5, "FWD": 3}
XI_MIN = {"GKP": 1, "DEF": 3, "MID": 2, "FWD": 1}
XI_MAX = {"GKP": 1, "DEF": 5, "MID": 5, "FWD": 3}

API_BASE = "https://fantasy.premierleague.com/api/"
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
    Also returns a short text list of opponents for display.
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


def normalise(series):
    """Scale a pandas Series to 0..1 (all equal -> 0.5)."""
    lo, hi = series.min(), series.max()
    if hi == lo:
        return pd.Series(0.5, index=series.index)
    return (series - lo) / (hi - lo)


def build_player_table(bootstrap, fixtures, horizon_gws):
    """One row per player with every raw and normalised metric the app needs."""
    events = bootstrap["events"]
    teams = {t["id"]: t for t in bootstrap["teams"]}
    target_gw = get_target_gw(events)
    finished_gws = count_finished_gws(events)

    team_form = compute_team_form(fixtures, CONFIG["team_form_matches"])
    ease_h, opp_h = compute_fixture_ease(fixtures, target_gw, horizon_gws)
    ease_1, opp_1 = compute_fixture_ease(fixtures, target_gw, 1)

    total_team_goals = {}
    rows = []
    for p in bootstrap["elements"]:
        team_id = p["team"]
        total_team_goals[team_id] = total_team_goals.get(team_id, 0) + p["goals_scored"]

    for p in bootstrap["elements"]:
        team_id = p["team"]
        team = teams[team_id]
        pos = POSITIONS[p["element_type"]]
        opp_list = opp_1.get(team_id, [])
        next_fixture = ", ".join(
            f"{teams[o]['short_name']} ({venue})" for o, venue in opp_list
        ) or "BLANK"
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
            "price": p["now_cost"] / 10.0,
            "price_raw": p["now_cost"],
            "form": float(p.get("form") or 0),
            "ep_next": float(p.get("ep_next") or 0),
            "ownership": float(p.get("selected_by_percent") or 0),
            "minutes": p.get("minutes", 0),
            "goals": p.get("goals_scored", 0),
            "assists": p.get("assists", 0),
            "status": p.get("status", "a"),
            "chance": p.get("chance_of_playing_next_round"),
            "team_form_raw": tf_raw,
            "strength_raw": strength,
            "ease_h_raw": ease_h.get(team_id, 0),
            "ease_1_raw": ease_1.get(team_id, 0),
            "fixtures_this_gw": len(opp_1.get(team_id, [])),
            "team_goals_total": total_team_goals.get(team_id, 0),
        })

    df = pd.DataFrame(rows)

    # Availability: fit, or doubtful with a >=75% chance of playing
    df["available"] = (df["status"] == "a") | (
        (df["status"] == "d") & (df["chance"].fillna(100) >= 75)
    )

    # Minutes gate only applies once we trust current-season data
    if finished_gws >= CONFIG["form_trust_gws"]:
        needed = 90 * finished_gws * CONFIG["min_minutes_share"]
        df["enough_minutes"] = df["minutes"] >= needed
    else:
        df["enough_minutes"] = True

    df["eligible"] = df["available"] & df["enough_minutes"]

    # Normalised components (computed across eligible players only, then
    # applied to everyone so the squad solver can still price up fringe picks)
    pool = df[df["eligible"]]
    for col, out in [
        ("form", "form_n"), ("ep_next", "ep_n"), ("price_raw", "price_n"),
        ("ease_h_raw", "ease_h_n"), ("ease_1_raw", "ease_1_n"),
    ]:
        df[out] = ((df[col] - pool[col].min()) /
                   ((pool[col].max() - pool[col].min()) or 1)).clip(0, 1)

    # Team form and strength are normalised within position groups, because
    # "goals conceded" numbers only make sense compared with other defences
    for group_cols in [("team_form_raw", "team_form_n"), ("strength_raw", "strength_n")]:
        raw, out = group_cols
        df[out] = 0.5
        for att in (True, False):
            mask = df["pos"].isin(["MID", "FWD"]) if att else df["pos"].isin(["GKP", "DEF"])
            sub = df.loc[mask & df["eligible"], raw]
            if len(sub) and sub.max() != sub.min():
                df.loc[mask, out] = ((df.loc[mask, raw] - sub.min()) /
                                     (sub.max() - sub.min())).clip(0, 1)

    # Involvement: share of the team's goals a player has scored or assisted
    df["involvement"] = df.apply(
        lambda r: (r["goals"] + r["assists"]) / r["team_goals_total"]
        if r["team_goals_total"] > 0 else 0.0, axis=1)
    inv_pool = df.loc[df["eligible"], "involvement"]
    rng = (inv_pool.max() - inv_pool.min()) or 1
    df["involvement_n"] = ((df["involvement"] - inv_pool.min()) / rng).clip(0, 1)

    # ------------------------------------------------------------------
    # Composite scores.
    # alpha slides from 0 (pre-season: trust pre-season signals) to 1
    # (trust real current-season form) over the first few gameweeks.
    # ------------------------------------------------------------------
    alpha = min(finished_gws / CONFIG["form_trust_gws"], 1.0)
    wm, we = CONFIG["weights_mature"], CONFIG["weights_early"]
    cm, ce = CONFIG["weights_captain_mature"], CONFIG["weights_captain_early"]

    mature_h = (wm["form"] * df["form_n"] + wm["fixture"] * df["ease_h_n"]
                + wm["team_form"] * df["team_form_n"] + wm["ep"] * df["ep_n"])
    early_h = (we["fixture"] * df["ease_h_n"] + we["strength"] * df["strength_n"]
               + we["price"] * df["price_n"] + we["ep"] * df["ep_n"])
    df["score_horizon"] = alpha * mature_h + (1 - alpha) * early_h

    mature_1 = (cm["form"] * df["form_n"] + cm["fixture"] * df["ease_1_n"]
                + cm["team_form"] * df["team_form_n"])
    early_1 = (ce["fixture"] * df["ease_1_n"] + ce["strength"] * df["strength_n"]
               + ce["price"] * df["price_n"])
    df["score_gw"] = alpha * mature_1 + (1 - alpha) * early_1

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
    cols = ["name", "team", "pos", "next_fixture", "price", "form", "ownership"] + (extra_cols or []) + [score_col]
    out = df[cols].rename(columns={
        "name": "Player", "team": "Team", "pos": "Pos", "price": "Price (£m)",
        "form": "Form", "ownership": "Owned %", "fixtures_this_gw": "Fixtures",
        "next_fixture": "Next",
        score_col: "Score",
    })
    out["Score"] = out["Score"].round(3)
    st.dataframe(out, hide_index=True, width="stretch")


def show_squad(squad, score_col):
    pos_order = {"GKP": 0, "DEF": 1, "MID": 2, "FWD": 3}
    squad = squad.sort_values(by=["starter", "pos", score_col],
                              key=lambda s: s.map(pos_order) if s.name == "pos" else s,
                              ascending=[False, True, False])
    starters = squad[squad["starter"]]
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
def main():
    st.set_page_config(page_title="FPL Ones to Watch", page_icon="⚽")
    st.title("⚽ FPL Ones to Watch")

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
         f"Optimised over the next {CONFIG['horizon_gws']} gameweeks."),
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
