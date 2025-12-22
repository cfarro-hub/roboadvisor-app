import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize
import datetime as dt
import matplotlib.pyplot as plt

# ===== Global config =====
RISK_FREE = 0.02
LOOKBACK_YEARS = 5

# ===== Page setup =====
st.set_page_config(page_title="Clyde – Robo‑Advisor", layout="wide")

# Global CSS for layout and questionnaire spacing
st.markdown(
    """
    <style>
    .block-container {
        max-width: 900px;
        padding-top: 3.5rem;
        padding-bottom: 2rem;
    }
    div[data-testid="stContainer"][class*="st-emotion-cache"][style*="border"]{
        border-radius: 1rem;
        border: 1px solid #0E9F6E;
        background: linear-gradient(135deg, #ECFDF5,#FFFFFF);
        box-shadow: 0 10px 25px rgba(14, 159, 110, 0.12);
        padding: 1.1rem 1.4rem;
        margin-bottom: 1.25rem;
    }
    form div[data-baseweb="base-input"] {
        margin-bottom: 0.25rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Track where user is: "landing" or "app"
if "page" not in st.session_state:
    st.session_state["page"] = "landing"


def go_to_app():
    st.session_state["page"] = "app"


def go_to_landing():
    st.session_state["page"] = "landing"


def reset_profile():
    for key in ["profile", "esg_only", "mu", "cov", "tickers",
                "base_table", "built_portfolios", "chosen_strategy",
                "invest_amount"]:
        if key in st.session_state:
            del st.session_state[key]


# ===== Landing page =====
if st.session_state["page"] == "landing":
    # Hero section
    hero_left, hero_right = st.columns([2, 1])
    with hero_left:
        st.title("Meet Clyde, your robo‑advisor")
        st.markdown(
            "Clyde helps you build and maintain a diversified portfolio "
            "based on your goals, time horizon, and comfort with risk."
        )
        st.markdown(
            "- Automated, diversified ETF portfolios\n"
            "- Built from your answers to a short questionnaire\n"
            "- Monitored and adjusted over time"
        )
        if st.button("📋 Start your questionnaire"):
            go_to_app()
            st.rerun()
    with hero_right:
        st.image("clyde.png", width=220)

    st.markdown("---")

    # How Clyde works
    st.markdown("### How Clyde works")

    step_cols = st.columns(3)
    steps = [
        ("1. Answer a few questions",
         "Share your goals, investment horizon, and comfort with market ups and downs."),
        ("2. Get a diversified portfolio",
         "Clyde suggests a mix of ETFs that fits your risk profile."),
        ("3. Stay on track automatically",
         "Your portfolio can be monitored and rebalanced over time."),
    ]
    for col, (title, text) in zip(step_cols, steps):
        with col:
            card = st.container(border=True)
            with card:
                st.subheader(title)
                st.write(text)

    st.markdown("---")

    # Who Clyde is for
    st.markdown("### Who Clyde might be right for")
    st.markdown(
        "- You are investing for 3 or more years.\n"
        "- You want a professionally designed ETF mix instead of picking single stocks.\n"
        "- You are comfortable with some risk in exchange for potential growth.\n"
        "- You like managing your account online."
    )

    st.stop()  # do not run the rest of the app while on landing page

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


def max_sharpe_with_risk_target(mu, cov, rf, max_vol):
    n = len(mu)
    x0 = equal_weight(n)
    bounds = [(0, 1)] * n
    cons = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1},
        {"type": "ineq", "fun": lambda w: max_vol - portfolio_vol(w, cov)},
    ]
    def obj(w): return -sharpe_ratio(w, mu, cov, rf)
    res = minimize(obj, x0, method="SLSQP", bounds=bounds, constraints=cons)
    return res.x


def efficient_frontier(mu, cov, rf=RISK_FREE, n_points=150):
    vols = []
    rets = []

    w_gmv = gmv_portfolio(cov)
    min_vol = portfolio_vol(w_gmv, cov)
    max_vol = min_vol * 3

    for target_vol in np.linspace(min_vol, max_vol, n_points):
        w = max_sharpe_with_risk_target(mu, cov, rf, target_vol)
        vols.append(portfolio_vol(w, cov))
        rets.append(portfolio_return(w, mu))

    return np.array(vols), np.array(rets)


# ===== Inflation advice helper =====
def inflation_advice(scenario, profile):
    if scenario == "Deflation (−1%)":
        msg = (
            "In deflation, cash and high-quality bonds usually hold up better, "
            "while risk assets like equities can suffer as growth slows. "
            "Keeping some cash and investment-grade bonds can help you ride out stress periods."
        )
        if profile == "aggressive":
            msg += " As an aggressive investor, avoid panic-selling but consider trimming very speculative positions."
        else:
            msg += " As a conservative or balanced investor, maintaining diversification and quality is key."
    elif scenario == "Low inflation (2%)":
        msg = (
            "Low, stable inflation is usually a good environment for diversified stock-bond portfolios. "
            "Your focus should stay on long-term goals rather than tactical changes."
        )
        msg += " Keeping your current risk profile and rebalancing occasionally is typically enough."
    elif scenario == "Moderate inflation (4%)":
        msg = (
            "Moderate inflation starts to erode bond and cash returns in real terms, "
            "but growth assets like equities can still compensate over time. "
            "Tilting slightly toward quality stocks, real assets, or shorter-duration bonds can help."
        )
        if profile == "conservative":
            msg += " As a conservative investor, review how much is in long-term fixed-rate bonds and consider shortening duration."
        else:
            msg += " As a balanced or aggressive investor, make sure you are not overly concentrated in long-duration bonds."
    else:  # "High inflation (7%)"
        msg = (
            "High inflation strongly reduces your real return; bonds and cash can lose purchasing power quickly. "
            "Historically, equities tied to real assets and inflation-linked securities have offered better protection."
        )
        if profile == "conservative":
            msg += " Consider spreading risk across short-term bonds, some equities, and possibly inflation-linked bonds instead of holding too much cash."
        else:
            msg += " As a growth-oriented investor, keep diversified equities but be mindful of very rate-sensitive sectors and very long-duration bonds."

    msg += " This is general guidance, not personalized financial advice."
    return msg


# ===== Risk profiling =====
def risk_profile_from_answers():
    st.markdown("Tell us about yourself")

    with st.form("risk_form"):
        st.markdown("#### Time horizon and risk")

        horizon = st.slider("Investment horizon (years)", 1, 30, 10)
        drawdown = st.slider("Maximum loss in one year you can tolerate (%)", 5, 50, 20)

        st.markdown("#### Your situation")

        income_stability = st.radio(
            "How stable is your income?",
            ["Unstable", "Somewhat stable", "Very stable"],
        )
        emergency = st.radio(
            "Do you have an emergency fund (3–6 months of expenses)?",
            ["No", "Yes"],
        )
        exp = st.radio(
            "How experienced are you with investing?",
            ["None", "Some", "Advanced"],
        )

        st.markdown("#### Goals and preferences")

        goal = st.radio(
            "What is your main goal?",
            ["Capital preservation", "Balanced growth", "Aggressive growth"],
        )
        behav = st.radio(
            "If your portfolio dropped 20%, what would you most likely do?",
            ["Sell everything", "Hold", "Buy more"],
        )
        esg_pref = st.radio(
            "Do you prefer ESG‑focused investments only?",
            ["No", "Yes"],
            horizontal=True,
        )

        submitted = st.form_submit_button("Continue to portfolio suggestions")

    if not submitted:
        return None, None

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

    esg_only = (esg_pref == "Yes")

    st.success(f"Inferred risk profile: {profile.upper()} (score={score})")

    st.session_state["profile"] = profile
    st.session_state["esg_only"] = esg_only

    return profile, esg_only


def risk_target_from_profile(profile, vol_ms):
    """Choose a target volatility relative to the unconstrained max‑Sharpe vol."""
    if profile == "conservative":
        return vol_ms * 0.7    # 30% less risk than max Sharpe
    elif profile == "balanced":
        return vol_ms          # same risk as max Sharpe
    elif profile == "aggressive":
        return vol_ms * 1.3    # 30% more risk than max Sharpe (if feasible)
    return vol_ms


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
ETF_LABELS = {
    # Core building blocks
    "VTI": "VTI – US Total Stock Market",
    "VEA": "VEA – Developed Ex‑US Equities",
    "VWO": "VWO – Emerging Markets Equities",
    "BND": "BND – US Total Bond Market",
    "BNDX": "BNDX – Intl Investment‑Grade Bonds (Hedged)",
    "VNQ": "VNQ – US REITs",
    "GLD": "GLD – Gold Trust",

    # Value Tilt
    "VTV": "VTV – US Large‑Cap Value",
    "VOE": "VOE – US Mid‑Cap Value",
    "VBR": "VBR – US Small‑Cap Value",

    # Innovative Technology
    "QQQ": "QQQ – Nasdaq‑100",
    "ARKK": "ARKK – Innovation ETF",
    "ICLN": "ICLN – Global Clean Energy",
    "SMH": "SMH – Semiconductor ETF",
    "BOTZ": "BOTZ – Robotics & AI",

    # ESG / Impact
    "ESGU": "ESGU – US ESG Equities",
    "ESGD": "ESGD – Dev. Mkts ESG Equities",
    "ESGE": "ESGE – EM ESG Equities",
    "ESGN": "ESGN – ESG Global Multifactor",
    "CRBN": "CRBN – Global Low‑Carbon Equities",
    "TAN": "TAN – Solar Energy ETF",
    "GRNB": "GRNB – Green Bond ETF",

    # Bonds / cash
    "AGG": "AGG – US Aggregate Bond",
    "LQD": "LQD – Inv‑Grade Corporate Bond",
    "HYG": "HYG – High‑Yield Corporate Bond",
    "BIL": "BIL – 1–3 Month T‑Bill ETF",

    # Smart beta
    "GSLC": "GSLC – US Large‑Cap Core Smart Beta",
    "GSEW": "GSEW – US Equal‑Weight",
    "QUAL": "QUAL – Quality Factor",
    "MTUM": "MTUM – Momentum Factor",

    # Crypto
    "IBIT": "IBIT – Bitcoin ETF",
    "FBTC": "FBTC – Bitcoin ETF",
    "ETHE": "ETHE – Ethereum ETF",
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


# ===== Main app (questionnaire + portfolios) =====
if st.session_state["page"] == "app":

    # Top bar + hero (Betterment-style)
    top_left, top_right = st.columns([0.6, 0.4])
    with top_left:
        back_col1, back_col2 = st.columns([0.25, 0.75])
        with back_col1:
            if st.button("← Home", use_container_width=True):
                go_to_landing()
                st.rerun()
        with back_col2:
            st.markdown("#### Clyde – Automated investing")
    with top_right:
        st.markdown(
            "<div style='text-align:right; font-size:0.9rem; color:#4B5563;'>"
            "Secure • Diversified • Low effort"
            "</div>",
            unsafe_allow_html=True,
        )

    hero_left, hero_right = st.columns([1.3, 1])
    with hero_left:
        st.markdown("## Start investing on autopilot")
        st.markdown(
            "Clyde builds and monitors a diversified ETF portfolio for you—"
            "based on your goals, time horizon, and comfort with risk."
        )
        st.markdown(
            "- Automated rebalancing and risk management\n"
            "- Globally diversified, low‑cost ETFs\n"
            "- Portfolios matched to your risk profile"
        )
    with hero_right:
        st.image("clyde.png", width=120)
        st.metric("Typical time to get a plan", "≈ 2 minutes")

    st.markdown("---")

    # Questionnaire + amount in two columns
   left_col, right_col = st.columns([1.1, 0.9])

    with left_col:
        st.markdown("## 1. Tell Clyde about yourself")
    
        # Button + slight spacer
        if st.button("Change my answers / risk profile"):
            reset_profile()
        st.write("")  # small vertical gap
    
        st.markdown("### Tell us about yourself")
    
        # Card around the questionnaire for consistent look
        left_card = st.container(border=True)
        with left_card:
            st.markdown("#### Time horizon and risk")
            # <-- keep all your sliders / radios exactly as before inside here
            profile, esg_only = (
                risk_profile_from_answers()
                if "profile" not in st.session_state or "esg_only" not in st.session_state
                else (st.session_state["profile"], st.session_state["esg_only"])
            )
    
    with right_col:
        # Heading OUTSIDE the card so it lines up with the left one
        st.markdown("## 2. Choose how much to invest")
    
        invest_card = st.container(border=True)
        with invest_card:
            st.markmarkdown("### Total investment amount")
            invest_amount = st.number_input(
                " ",  # empty string to hide the label
                min_value=500,
                max_value=1_000_000,
                value=10_000,
                step=1000,
                label_visibility="collapsed",
            )
            st.session_state["invest_amount"] = invest_amount
            st.caption("Minimum investment amount is €500.")


    st.markdown("---")

    # Flag to remember if portfolios were built
    if "built_portfolios" not in st.session_state:
        st.session_state["built_portfolios"] = False

    with st.expander("View your recommended portfolios", expanded=True):
        st.markdown("### Clyde’s suggested strategies")
        st.caption(
            "These portfolios are built from ETF mixes that match your risk profile. "
            "Pick one to see its details and risk–return profile."
        )

        candidate_names = strategies_for_profile(profile, esg_only)

        # init state
        if "mu" not in st.session_state:
            st.session_state["mu"] = None
            st.session_state["cov"] = None
            st.session_state["tickers"] = None
            st.session_state["base_table"] = None

        chosen_strategy = st.session_state.get("chosen_strategy", candidate_names[0])

        cards_per_row = 3
        for row_start in range(0, len(candidate_names), cards_per_row):
            row_names = candidate_names[row_start:row_start + cards_per_row]
            cols = st.columns(len(row_names))
            for col, name in zip(cols, row_names):
                info = STRATEGIES[name]
                with col:
                    label = name + " 🌱" if info["type"] == "esg" else name
                    card = st.container(border=True)
                    with card:
                        st.subheader(label)
                        st.caption(info["description"])
                        st.markdown(
                            f"<span style='font-size:0.8rem; color:#6B7280;'>"
                            f"Type: {info['type'].replace('-', ' ').title()}</span>",
                            unsafe_allow_html=True,
                        )
                        if st.button("Select", key=f"choose_{name}"):
                            chosen_strategy = name

        st.session_state["chosen_strategy"] = chosen_strategy

        st.markdown(f"### Selected portfolio: **{chosen_strategy}**")
        st.write(STRATEGIES[chosen_strategy]["description"])

        tickers = STRATEGIES[chosen_strategy]["tickers"]
        universe_labels = [ETF_LABELS.get(t, t) for t in tickers]
        st.write("ETF universe:")
        for lbl in universe_labels:
            st.write(f"- {lbl}")

        st.info(
            "Clyde builds multiple versions of this portfolio (equal‑weight, minimum‑risk, "
            "max Sharpe, and a version matched to your risk profile). You can compare them "
            "and then customize the weights."
        )

        # Build portfolios
        if st.button("Build portfolios for this strategy"):
            with st.spinner("Downloading prices and running optimizations…"):
                prices = get_price_data(tickers, years=LOOKBACK_YEARS)
                mu, cov, tickers_new = compute_returns_and_cov(prices)

            st.session_state["mu"] = mu
            st.session_state["cov"] = cov
            st.session_state["tickers"] = tickers_new

            n = len(tickers_new)
            ew = equal_weight(n)
            gmv = gmv_portfolio(cov)
            ms = max_sharpe_portfolio(mu, cov, RISK_FREE)

            # Volatility of the unconstrained Max Sharpe portfolio
            vol_ms = portfolio_vol(ms, cov)

            # Profile‑specific target volatility
            target_vol = risk_target_from_profile(profile, vol_ms)

            ms_prof = max_sharpe_with_risk_target(
                mu, cov, RISK_FREE, target_vol
            )

            def row(name, w):
                return {
                    "Portfolio": name,
                    "Expected Return": portfolio_return(w, mu),
                    "Volatility": portfolio_vol(w, cov),
                    "Sharpe": sharpe_ratio(w, mu, cov, RISK_FREE),
                    **{f"w_{t}": w[i] for i, t in enumerate(tickers_new)}
                }

            st.session_state["base_table"] = pd.DataFrame([
                row("Equal Weight", ew),
                row("Global Min Variance (GMV)", gmv),
                row("Max Sharpe", ms),
                row(f"Profile Max Sharpe ({profile})", ms_prof),
            ])
            st.session_state["built_portfolios"] = True

        # Show base portfolios table and tools
        if (
            st.session_state.get("built_portfolios", False)
            and st.session_state["mu"] is not None
            and st.session_state["cov"] is not None
            and st.session_state["base_table"] is not None
        ):
            mu = st.session_state["mu"]
            cov = st.session_state["cov"]
            tickers = st.session_state["tickers"]
            base_table = st.session_state["base_table"]

            st.subheader("Base portfolios for this strategy")
            st.dataframe(
                base_table.style.format(
                    {"Expected Return": "{:.2%}",
                     "Volatility": "{:.2%}",
                     "Sharpe": "{:.2f}"} |
                    {c: "{:.1%}" for c in base_table.columns if c.startswith("w_")}
                )
            )

            # === KPI summary for the currently selected base portfolio ===
            options = list(base_table["Portfolio"])
            if profile == "conservative":
                safe_mask = base_table["Volatility"] <= 0.15   # 15% vol cut-off; tweak as needed
                safe_names = base_table.loc[safe_mask, "Portfolio"].tolist()
                if safe_names:
                    options = safe_names
            ref_name = st.selectbox("Choose base portfolio", options)
            base_row = base_table[base_table["Portfolio"] == ref_name].iloc[0]
            base_sh = base_row["Sharpe"]

            st.markdown("#### Snapshot of selected portfolio")

            col_r, col_v, col_s = st.columns(3)
            with col_r:
                st.metric("Expected return", f"{base_row['Expected Return']:.2%}")
            with col_v:
                st.metric("Volatility", f"{base_row['Volatility']:.2%}")
            with col_s:
                st.metric("Sharpe ratio", f"{base_sh:.2f}")

            # Inflation / deflation scenarios
            st.markdown("#### Inflation / deflation scenarios")

            infl_scenario = st.selectbox(
                "Choose an inflation scenario",
                ["Deflation (−1%)", "Low inflation (2%)", "Moderate inflation (4%)", "High inflation (7%)"],
            )

            scenario_map = {
                "Deflation (−1%)": -0.01,
                "Low inflation (2%)": 0.02,
                "Moderate inflation (4%)": 0.04,
                "High inflation (7%)": 0.07,
            }
            pi = scenario_map[infl_scenario]

            nominal_ret = float(base_row["Expected Return"])
            real_ret = nominal_ret - pi

            col_nom, col_real = st.columns(2)
            with col_nom:
                st.metric("Nominal expected return", f"{nominal_ret:.2%}")
            with col_real:
                st.metric("Real return (after inflation)", f"{real_ret:.2%}")

            # Clyde's narrative advice
            st.markdown("#### Clyde’s take on this scenario")
            st.info(inflation_advice(infl_scenario, profile))

            st.markdown("#### Clyde’s portfolio strategy suggestion")

            def recommend_strategy_for_inflation(scenario, profile, strategies_for_user):
                """
                strategies_for_user = list of strategy names already filtered
                by profile/ESG (your candidate_names).
                Returns (strategy_name, message).
                """

                def pick(preferences):
                    for s in preferences:
                        if s in strategies_for_user:
                            return s
                    return strategies_for_user[0]

                if scenario == "Deflation (−1%)":
                    prefs = ["Cash Reserve", "BlackRock Target Income", "Core"]
                    strat = pick(prefs)
                    msg = (
                        "In deflation, preserving capital matters most. I would lean toward more defensive, "
                        "income-focused strategies like cash reserves and high-quality bond portfolios, "
                        "using Core mainly as a diversifier rather than the main growth engine."
                    )

                elif scenario == "Low inflation (2%)":
                    prefs = ["Core", "ESG Balanced", "Value Tilt"]
                    strat = pick(prefs)
                    msg = (
                        "At roughly 2% inflation, diversified portfolios tend to work well. I would anchor you "
                        "in a Core or ESG Balanced strategy that matches your risk profile and simply rebalance "
                        "periodically instead of making big tactical changes."
                    )

                elif scenario == "Moderate inflation (4%)":
                    if profile == "conservative":
                        prefs = ["Core", "ESG Balanced", "BlackRock Target Income"]
                    else:
                        prefs = ["Value Tilt", "Core", "ESG Global Equity"]
                    strat = pick(prefs)
                    msg = (
                        "Around 4% inflation, bond and cash returns start to be eroded in real terms. I would still "
                        "use Core or ESG Balanced as your anchor, but tilt more toward value and real-economy "
                        "equity strategies, and avoid concentrating too much in very long-maturity bonds."
                    )

                else:  # High inflation (7%)
                    if profile == "conservative":
                        prefs = ["Core", "ESG Balanced", "BlackRock Target Income"]
                    elif profile == "balanced":
                        prefs = ["Core", "Value Tilt", "ESG Global Equity", "Climate Impact"]
                    else:
                        prefs = ["Value Tilt", "Innovative Technology", "Crypto ETF", "Climate Impact"]
                    strat = pick(prefs)
                    msg = (
                        "With high inflation, protecting your purchasing power matters more than maximizing nominal yield. "
                        "I would favor strategies with more equity and real-asset exposure, and use nominal bonds mainly "
                        "for diversification rather than as the primary return driver."
                    )

                msg += " This is general guidance about strategy types, not personalized investment advice."
                return strat, msg

            candidate_names = strategies_for_profile(profile, esg_only)
            rec_strategy, rec_msg = recommend_strategy_for_inflation(
                infl_scenario, profile, candidate_names
            )

            st.write(f"**Suggested portfolio:** {rec_strategy}")
            st.caption(rec_msg)

            st.subheader("Adjust weights and see the impact")

            st.write(
                f"Base {ref_name}: return {base_row['Expected Return']:.2%}, "
                f"vol {base_row['Volatility']:.2%}, Sharpe {base_sh:.2f}"
            )

            # Sliders in percent; will be normalized to sum 100%
            weight_cols = st.columns(len(tickers))
            raw_weights = []
            for i, t in enumerate(tickers):
                default_w = float(base_row[f"w_{t}"]) * 100.0
                with weight_cols[i]:
                    w_pct = st.slider(f"{t} (%)", 0.0, 100.0, default_w)
                raw_weights.append(w_pct)

            if st.button("Evaluate custom portfolio"):
                raw = np.array(raw_weights)

                if raw.sum() == 0:
                    w = equal_weight(len(tickers))
                else:
                    w = raw / raw.sum()

                ret_new = portfolio_return(w, mu)
                vol_new = portfolio_vol(w, cov)
                sh_new = sharpe_ratio(w, mu, cov, RISK_FREE)

                st.write("New weights (normalized to 100%):")
                labels = [ETF_LABELS.get(t, t) for t in tickers]
                st.dataframe(
                    pd.DataFrame({"Ticker": tickers, "Name": labels, "Weight": w})
                      .style.format({"Weight": "{:.1%}"})
                )

                total_invest = st.session_state.get("invest_amount", 0)
                allocation = w * total_invest
                st.write("Investment amounts for each ETF at your chosen total investment:")
                st.dataframe(
                    pd.DataFrame(
                        {
                            "Ticker": tickers,
                            "Name": labels,
                            "Weight": w,
                            "Amount": allocation,
                        }
                    ).style.format({"Weight": "{:.1%}", "Amount": "€{:,.0f}"})
                )


                st.write(
                    f"New portfolio: return {ret_new:.2%}, "
                    f"vol {vol_new:.2%}, Sharpe {sh_new:.2f}"
                )

                st.info(roboadvisor_comment(ret_new, vol_new, sh_new, base_sh, profile))

                # === Efficient frontier chart ===
                vols, rets = efficient_frontier(mu, cov, rf=RISK_FREE, n_points=150)

                fig, ax = plt.subplots(figsize=(7, 4))

                ax.plot(
                    vols,
                    rets,
                    label="Efficient frontier",
                    color="#4ade80",
                    linewidth=2.0,
                )

                ax.scatter(
                    base_row["Volatility"],
                    base_row["Expected Return"],
                    color="#22c55e",
                    edgecolor="white",
                    s=50,
                    zorder=3,
                    label=f"Base: {ref_name}",
                )

                ax.scatter(
                    vol_new,
                    ret_new,
                    color="#f97316",
                    edgecolor="white",
                    marker="*",
                    s=80,
                    zorder=4,
                    label="Your portfolio",
                )

                ax.set_title("Efficient Frontier", fontsize=16, pad=10)
                ax.set_xlabel("Volatility (%)", fontsize=12)
                ax.set_ylabel("Return (%)", fontsize=12)

                ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x*100:.0f}%"))
                ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y*100:.0f}%"))

                all_vols = np.concatenate([vols, [base_row["Volatility"], vol_new]])
                all_rets = np.concatenate([rets, [base_row["Expected Return"], ret_new]])

                x_min = all_vols.min() * 0.9
                x_max = all_vols.max() * 1.1
                y_min = all_rets.min() * 0.9
                y_max = all_rets.max() * 1.1

                ax.set_xlim(x_min, x_max)
                ax.set_ylim(y_min, y_max)

                ax.grid(True, which="both", linestyle="--", alpha=0.25)
                ax.legend(loc="best", frameon=False)

                plt.tight_layout()
                st.pyplot(fig)

            st.markdown("---")
            st.markdown("### How Clyde manages this portfolio")
            st.markdown(
                "- **Automated rebalancing:** When markets move, Clyde would rebalance toward your target mix.\n"
                "- **Risk‑based design:** Portfolios are optimized for return per unit of risk, not just raw return.\n"
                "- **Goal‑based:** Over time you’ll be able to link portfolios to specific goals (retirement, house, etc.)."
            )

        else:
            st.info("Click 'Build portfolios for this strategy' to see portfolio options.")
