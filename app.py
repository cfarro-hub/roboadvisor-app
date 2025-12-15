import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize
import datetime as dt

# ===== Global config =====
RISK_FREE = 0.02
LOOKBACK_YEARS = 5

DEFAULT_TICKERS = ["SPY", "VEA", "EEM", "AGG", "BNDX", "VNQ", "GLD"]
# ===== Data + math helpers =====
def get_price_data(tickers, years=5):
    end = dt.date.today()
    start = end - dt.timedelta(days=365 * years)
    data = yf.download(tickers, start=start, end=end, auto_adjust=True)

    if isinstance(data.columns, pd.MultiIndex):
        if "Adj Close" in data.columns.get_level_values(0):
            data = data["Adj Close"]
        elif "Close" in data.columns.get_level_values(0):
            data = data["Close"]
        else:
            raise ValueError("No Adj Close or Close in downloaded data.")
    else:
        if "Adj Close" in data.columns:
            data = data["Adj Close"]
        elif "Close" in data.columns:
            data = data["Close"]
        else:
            raise ValueError(f"No Adj Close or Close. Columns: {data.columns}")

    data = data.dropna(how="all")
    if isinstance(data, pd.Series):
        data = data.to_frame()
    return data


def compute_returns_and_cov(prices):
    rets = prices.resample("M").last().pct_change().dropna()
    mu = rets.mean() * 12
    cov = rets.cov() * 12
    return mu.values, cov.values, list(mu.index)


def portfolio_return(w, mu):
    return float(np.dot(w, mu))


def portfolio_vol(w, cov):
    return float(np.sqrt(w.T @ cov @ w))


def sharpe_ratio(w, mu, cov, rf=RISK_FREE):
    r = portfolio_return(w, mu)
    v = portfolio_vol(w, cov)
    return 0.0 if v == 0 else (r - rf) / v


def equal_weight(n):
    return np.ones(n) / n


def gmv_portfolio(cov):
    n = cov.shape[0]
    x0 = equal_weight(n)
    bounds = [(0, 1)] * n
    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    res = minimize(lambda w: portfolio_vol(w, cov), x0, method="SLSQP",
                   bounds=bounds, constraints=cons)
    return res.x


def max_sharpe_portfolio(mu, cov, rf=RISK_FREE):
    n = len(mu)
    x0 = equal_weight(n)
    bounds = [(0, 1)] * n
    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    def obj(w): return -sharpe_ratio(w, mu, cov, rf)
    res = minimize(obj, x0, method="SLSQP", bounds=bounds, constraints=cons)
    return res.x


# ===== Risk profiling =====
def risk_profile_from_answers():
    st.sidebar.subheader("Risk questionnaire")

    horizon = st.sidebar.slider("Investment horizon (years)", 1, 30, 10)
    income_stability = st.sidebar.radio(
        "Income stability",
        ["Unstable", "Somewhat stable", "Very stable"]
    )
    emergency = st.sidebar.radio(
        "Emergency fund (3–6 months saved)?",
        ["No", "Yes"]
    )
    goal = st.sidebar.radio(
        "Main goal",
        ["Capital preservation", "Balanced growth", "Aggressive growth"]
    )
    drawdown = st.sidebar.slider(
        "Max loss in one year you tolerate (%)", 5, 50, 20
    )
    behav = st.sidebar.radio(
        "If portfolio drops 20%, you…",
        ["Sell everything", "Hold", "Buy more"]
    )
    exp = st.sidebar.radio(
        "Experience with investing",
        ["None", "Some", "Advanced"]
    )
    esg_pref = st.sidebar.radio(
        "How important is ESG to you?",
        ["No", "Yes"]
    )

    score = 0
    score += 0 if horizon < 3 else (1 if horizon < 7 else 2)
    score += {"Unstable": 0, "Somewhat stable": 1, "Very stable": 2}[income_stability]
    score += 0 if emergency == "No" else 1
    score += {"Capital preservation": 0, "Balanced growth": 1, "Aggressive growth": 2}[goal]
    score += 0 if drawdown < 10 else (1 if drawdown < 25 else 2)
    score += {"Sell everything": 0, "Hold": 1, "Buy more": 2}[behav]
    score += {"None": 0, "Some": 1, "Advanced": 2}[exp]

    if score <= 5:
        profile = "conservative"
    elif score <= 10:
        profile = "balanced"
    else:
        profile = "aggressive"

    st.sidebar.markdown(f"**Inferred risk profile:** {profile.upper()} (score={score})")

    esg_only = (esg_pref == "Yes")
    return profile, esg_only

def risk_target_from_profile(profile):
    if profile == "conservative":
        return 0.08
    elif profile == "balanced":
        return 0.15
    elif profile == "aggressive":
        return 0.25
    return 0.15


def max_sharpe_with_risk_target(mu, cov, rf, max_vol):
    n = len(mu)
    x0 = equal_weight(n)
    bounds = [(0, 1)] * n
    cons = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1},
        {"type": "ineq", "fun": lambda w: max_vol - portfolio_vol(w, cov)}
    ]
    def obj(w): return -sharpe_ratio(w, mu, cov, rf)
    res = minimize(obj, x0, method="SLSQP", bounds=bounds, constraints=cons)
    return res.x


# ===== Strategy definitions =====
STRATEGIES = {
    "Core": {
        "description": "Well-diversified, low-cost, global stock and bond ETFs.",
        "tickers": ["VTI", "VEA", "VWO", "BND", "BNDX"],
        "type": "mixed",
    },
    "Value Tilt": {
        "description": "Global portfolio tilted toward undervalued value stocks.",
        "tickers": ["VTV", "VOE", "VEA", "VBR", "VWO"],
        "type": "equity-heavy",
    },
    "Innovative Technology": {
        "description": "High-growth sectors: tech, clean energy, semis, robotics, etc.",
        "tickers": ["QQQ", "ARKK", "ICLN", "SMH", "BOTZ"],
        "type": "high-risk",
    },
    "Broad Impact": {
        "description": "Broad ESG portfolio screening for environmental and social factors.",
        "tickers": ["ESGU", "ESGD", "ESGE", "ESGN"],
        "type": "esg",
    },
    "Climate Impact": {
        "description": "Lower carbon emissions and green-project ETFs.",
        "tickers": ["CRBN", "ICLN", "TAN", "GRNB"],
        "type": "esg",
    },
    "Cash Reserve": {
        "description": "Cash-like exposure (short-term Treasuries / money market).",
        "tickers": ["BIL"],
        "type": "cash",
    },
    "BlackRock Target Income": {
        "description": "100% bond portfolio targeting income with lower equity risk.",
        "tickers": ["AGG", "LQD", "HYG"],
        "type": "bond-only",
    },
    "Goldman Sachs Smart Beta": {
        "description": "Smart beta factor ETFs seeking long-term outperformance.",
        "tickers": ["GSLC", "GSEW", "QUAL", "MTUM"],
        "type": "equity-heavy",
    },
    "Crypto ETF": {
        "description": "Exposure to Bitcoin and Ethereum via ETFs.",
        "tickers": ["IBIT", "FBTC", "ETHE"],
        "type": "high-risk",
    },
    "ESG Global Equity": {
        "description": "Global stock portfolio using broad ESG‑screened equity ETFs.",
        "tickers": ["ESGU", "ESGD", "ESGE"],
        "type": "esg",
    },
    "ESG Climate Leaders": {
        "description": "Climate‑focused ETFs: clean energy, low‑carbon and green bonds.",
        "tickers": ["ICLN", "CRBN", "TAN", "GRNB"],
        "type": "esg",
    },
    "ESG Balanced": {
        "description": "Balanced ESG mix of stocks and bonds.",
        "tickers": ["ESGU", "ESGD", "ESGE", "AGG"],
        "type": "esg",
    },
}


def strategies_for_profile(profile, esg_only: bool):
    esg_strats = [
        "Broad Impact",
        "ESG Global Equity",
        "ESG Climate Leaders",
        "ESG Balanced",
    ]

    non_esg_strats = [
        "Core",
        "Value Tilt",
        "Innovative Technology",
        "Cash Reserve",
        "BlackRock Target Income",
        "Goldman Sachs Smart Beta",
        "Crypto ETF",
    ]

    if esg_only:
        base = esg_strats
    else:
        base = non_esg_strats

    if profile == "conservative":
        base = [s for s in base if s not in ["Innovative Technology", "Crypto ETF"]]

    return base


def roboadvisor_comment(ret, vol, sh, base_sh, profile):
    msgs = []
    delta = sh - base_sh
    if delta > 0.05:
        msgs.append("Sharpe ratio is higher than the reference: more efficient risk–return.")
    elif delta < -0.05:
        msgs.append("Sharpe ratio is lower than the reference: less efficient than the base portfolio.")
    else:
        msgs.append("Sharpe ratio is similar to the reference: change is small in efficiency terms.")
    if profile == "conservative" and vol > 0.12:
        msgs.append("Warning: volatility is high for a conservative profile.")
    if profile == "aggressive" and vol < 0.10:
        msgs.append("Note: volatility is low for an aggressive profile; you may be under‑risked.")
    return " ".join(msgs)


# ===== App layout =====
st.set_page_config(page_title="Robo-Advisor Demo", layout="wide")
st.title("Robo‑Advisor – Portfolio Strategies")

profile, esg_only = risk_profile_from_answers()
