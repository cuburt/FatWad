"""Risk-profile + allocation logic. Stays here (not in agents/) because the
buy_list and rebalance signals are pure functions, no LLM in the loop."""

ALLOCATION = {
    "Aggressive": {"Equity": 0.55, "Crypto": 0.20, "Real Estate": 0.15, "Cash": 0.10},
    "Balanced":   {"Equity": 0.50, "Crypto": 0.10, "Real Estate": 0.20, "Cash": 0.20},
    "Defensive":  {"Equity": 0.35, "Crypto": 0.05, "Real Estate": 0.25, "Cash": 0.35},
}

CASHFLOW_TYPES = {
    "Equity": True, "Real Estate": True, "Cash": True,
    "Crypto": False, "Physical": False,
}

RISK_QUESTIONS = [
    {"q": "Your portfolio drops 30% in a month. You:",
     "options": [("Buy more aggressively.", 3), ("Hold and wait.", 2),
                 ("Sell to stop the bleeding.", 0)]},
    {"q": "Time horizon for this money:",
     "options": [("20+ years.", 3), ("5-20 years.", 2), ("Under 5 years.", 0)]},
    {"q": "Preferred volatility:",
     "options": [("Don't care, give me upside.", 3), ("Some swings are fine.", 2),
                 ("Smooth and steady.", 0)]},
    {"q": "Income stability:",
     "options": [("Rock solid.", 3), ("Variable but reliable.", 2),
                 ("Lumpy / uncertain.", 0)]},
    {"q": "Net worth relative to annual burn:",
     "options": [("10x or more.", 3), ("3-10x.", 2), ("Under 3x.", 0)]},
]


def score_to_profile(score: int) -> str:
    if score >= 12:
        return "Aggressive"
    if score >= 7:
        return "Balanced"
    return "Defensive"


def buy_list(surplus: float, profile: str) -> dict[str, float]:
    if surplus <= 0 or profile not in ALLOCATION:
        return {}
    return {k: round(surplus * w, 2) for k, w in ALLOCATION[profile].items()}


def rebalance_signals(assets: list[dict], profile: str) -> list[str]:
    if profile not in ALLOCATION:
        return []
    weights = ALLOCATION[profile]
    totals: dict[str, float] = {}
    for a in assets:
        t = a.get("type", "Cash")
        totals[t] = totals.get(t, 0.0) + float(a.get("current_value", 0) or 0)
    nw = sum(totals.values())
    if nw <= 0:
        return []
    msgs: list[str] = []
    for t, target_w in weights.items():
        actual_w = totals.get(t, 0.0) / nw
        delta = actual_w - target_w
        if delta > 0.10:
            msgs.append(f"TRIM {t}: {actual_w*100:.1f}% of NW vs target {target_w*100:.0f}%.")
        elif delta < -0.10:
            msgs.append(f"ADD {t}: {actual_w*100:.1f}% of NW vs target {target_w*100:.0f}%.")
    return msgs
