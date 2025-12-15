import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize
import datetime as dt

# ===== Global config =====
RISK_FREE = 0.02
LOOKBACK_YEARS = 5

# ===== Clyde front page =====
st.set_page_config(
    page_title="Robo-Advisor Demo",
    layout="wide"
)

# Track whether user is still on intro page
if "show_intro" not in st.session_state:
    st.session_state["show_intro"] = True

# Intro page with big Clyde
if st.session_state["show_intro"]:
    st.title("Clyde – Your Robo‑Advisor")

    # Big Clyde image centered
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image("clyde.png", width=300)

    st.markdown("### 🤖 Hi, my name is Clyde.")
    st.write("I am your robo‑advisor. I will help you build a portfolio tailored to you.")
    st.write("When you are ready, click the button below to start the questionnaire.")

    if st.button("Please click here"):
        st.session_state["show_intro"] = False
        st.rerun()

    st.stop()  # stop app while on intro page

# Mini Clyde header (shown on all pages after intro)
header_col1, header_col2 = st.columns([0.1, 0.9])
with header_col1:
    st.image("clyde.png", width=60)
with header_col2:
    st.markdown("### Clyde – Your Robo‑Advisor")

# flag to remember if portfolios were built
if "built_portfolios" not in st.session_state:
    st.session_state["built_portfolios"] = False

DEFAULT_TICKERS = ["SPY", "VEA", "EEM", "AGG", "BNDX", "VNQ", "GLD"]

# ===== Data + math helpers =====
def get_price_data(tickers, years=5):
    end = dt.date.today()
    start = end - dt.timedelta(days=365 * years)
    data = yf.download(tickers, start=start, end=end, auto_adjust=True)

    if isinstance(data.columns, pd.MultiIndex):
        if "Adj Close" in data.columns
