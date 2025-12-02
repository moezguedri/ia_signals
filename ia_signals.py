from __future__ import annotations

import argparse
import os
import smtplib
from datetime import datetime, timezone
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Optional, Dict, List, Tuple

import certifi
import pandas as pd
import requests
import urllib3
import yfinance as yf

# ==========================
# SSL / CERTIFICATE SETUP
# ==========================

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

CA_BUNDLE = certifi.where()
os.environ["SSL_CERT_FILE"] = CA_BUNDLE
os.environ["REQUESTS_CA_BUNDLE"] = CA_BUNDLE
os.environ["CURL_CA_BUNDLE"] = CA_BUNDLE

# ==========================
# CONFIGURATION
# ==========================

# Full universe of tickers to display and monitor
TICKERS: List[str] = [
    "NVDA",
    "MSFT",
    "GOOGL",
    "META",
    "AMZN",
    "ASML",
    # "AMD",
    # "CCJ",
    # "ETN",
    # "VRT",
    "COST",
    # "QQQ",
]

# Core AI tickers used for allocation (internal)
CORE_AI_TICKERS = ["NVDA", "MSFT", "GOOGL", "META", "AMZN", "ASML", "AMD"]

# Tickers shown only for information (no effect on allocation)
INFO_ONLY_TICKERS = ["COST", "QQQ"]

# Thresholds applied to average of 10/30/90-day returns
PANIC_THRESHOLD = -0.08   # -8%
EUPHORIA_THRESHOLD = 0.10 # +10%

# Data consistency tolerance between Yahoo and Twelve Data
MISMATCH_TOLERANCE = 0.01

# Default monthly budget in USD for AI allocation
DEFAULT_BUDGET = 4000.0

# Signal weights for allocation
WEIGHTS = {
    "PANIC_BUY": 2.0,
    "NORMAL": 1.0,
    "EUPHORIA": 0.25,
}

# Reference monthly amounts in IBKR (used only for email suggestions)
IBKR_REFERENCE_MONTHLY: Dict[str, float] = {
    "NVDA": 1000.0,
    "MSFT": 700.0,
    "GOOGL": 700.0,
    "META": 500.0,
    "AMZN": 500.0,
    "ASML": 300.0,
    "COST": 100.0,
    # "AMD": 0.0,
    # "QQQ": 0.0,
}

# Target long-term portfolio allocation (in % of total portfolio)
PORTFOLIO_TARGETS: Dict[str, int] = {
    "NVDA": 25,
    "MSFT": 23,
    "GOOGL": 13,
    "META": 11,
    "AMZN": 11,
    "ASML": 12,
    "COST": 5,
}

# ==========================
# DATA SOURCES
# ==========================

def download_yahoo_close(ticker: str, lookback_days: int = 10) -> Optional[pd.Series]:
    """Download daily close prices from Yahoo Finance. Returns Series or None."""
    try:
        hist = yf.download(
            ticker,
            period=f"{lookback_days}d",
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=False,
        )
    except Exception as e:
        print(f"[Yahoo] ERROR for {ticker}: {e}")
        return None

    if not isinstance(hist, pd.DataFrame):
        print(f"[Yahoo] Invalid response type for {ticker}: {type(hist)}")
        return None

    if hist.empty:
        print(f"[Yahoo] Empty DataFrame for {ticker}")
        return None

    if "Close" not in hist.columns:
        print(f"[Yahoo] Missing 'Close' column for {ticker}: columns={list(hist.columns)}")
        return None

    close = hist["Close"]

    if isinstance(close, pd.DataFrame):
        # MultiIndex columns (field, ticker) or multiple columns
        if isinstance(close.columns, pd.MultiIndex):
            try:
                close = close.xs(ticker, axis=1, level=-1)
            except Exception:
                close = close.iloc[:, 0]
        else:
            close = close.iloc[:, 0]

    if not isinstance(close, pd.Series):
        print(f"[Yahoo] Close is not Series for {ticker}, type={type(close)}")
        return None

    close = close.dropna()

    if len(close) < 6:
        print(f"[Yahoo] Not enough data for {ticker} (n={len(close)})")
        return None

    return close


def download_twelve_close(ticker: str, lookback_days: int = 10) -> Optional[pd.Series]:
    """Download daily close prices from Twelve Data. Returns Series or None."""
    api_key = os.getenv("TWELVE_API_KEY")
    if not api_key:
        print("[TwelveData] Missing API key (TWELVE_API_KEY).")
        return None

    url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": ticker,
        "interval": "1day",
        "outputsize": lookback_days + 10,
        "apikey": api_key,
        "format": "JSON",
    }

    try:
        resp = requests.get(url, params=params, timeout=10, verify=CA_BUNDLE)
        data = resp.json()
    except Exception as e:
        print(f"[TwelveData] ERROR {ticker}: {e}")
        return None

    if "values" not in data:
        print(f"[TwelveData] No data for {ticker}: {data}")
        return None

    df = pd.DataFrame(data["values"])

    if "close" not in df.columns:
        print(f"[TwelveData] Invalid payload for {ticker}")
        return None

    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").set_index("datetime")

    series = df["close"].dropna()

    if len(series) < 6:
        return None

    return series.tail(lookback_days + 2)

# ==========================
# MULTI-HORIZON RETURNS
# ==========================

def compute_return(series: pd.Series, days: int) -> float:
    """
    Compute cumulative return over N trading days:
    (last / close_N_days_ago - 1)
    """
    series = series.dropna()
    if len(series) < days + 1:
        raise ValueError(f"Not enough data for {days}-day return")
    last = float(series.iloc[-1])
    prev = float(series.iloc[-(days + 1)])
    return (last / prev) - 1.0


def compute_multi_returns(series: pd.Series) -> Dict[str, float]:
    """
    Compute 10d, 30d, 90d returns.
    """
    return {
        "R10": compute_return(series, 10),
        "R30": compute_return(series, 30),
        "R90": compute_return(series, 90),
    }


def get_returns_robust(ticker: str) -> Tuple[Optional[Dict[str, float]], str, Optional[Dict[str, float]]]:
    """
    Return robust multi-horizon returns using Yahoo + Twelve Data.
    """
    yahoo = download_yahoo_close(ticker, lookback_days=120)
    twelve = download_twelve_close(ticker, lookback_days=120)

    yahoo_ret = None
    twelve_ret = None

    if yahoo is not None:
        try:
            yahoo_ret = compute_multi_returns(yahoo)
        except Exception as e:
            print(f"[Yahoo] Return computation failed for {ticker}: {e}")

    if twelve is not None:
        try:
            twelve_ret = compute_multi_returns(twelve)
        except Exception as e:
            print(f"[TwelveData] Return computation failed for {ticker}: {e}")

    if yahoo_ret is None and twelve_ret is None:
        return None, "NONE", None

    if yahoo_ret is not None and twelve_ret is None:
        return yahoo_ret, "YAHOO", None

    if yahoo_ret is None and twelve_ret is not None:
        return twelve_ret, "TWELVEDATA", None

    # Both available: check consistency across horizons
    diff = max(abs(yahoo_ret[k] - twelve_ret[k]) for k in yahoo_ret.keys())
    if diff > MISMATCH_TOLERANCE:
        print(f"[MISMATCH] {ticker}: Yahoo vs TwelveData return mismatch")
        return yahoo_ret, "MISMATCH", twelve_ret

    return yahoo_ret, "YAHOO", twelve_ret

# ==========================
# PRICE, TARGET, ATH
# ==========================

def get_price_and_target(ticker: str) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Fetch last price and analyst target from Yahoo (via yfinance).
    Returns (price, target, delta_pct) where delta_pct = (price/target - 1)*100.
    """
    price = None
    target = None
    delta_pct = None

    try:
        tk = yf.Ticker(ticker)

        # Price
        try:
            fast = getattr(tk, "fast_info", None)
            if fast:
                if "lastPrice" in fast:
                    price = float(fast["lastPrice"])
                elif "last_price" in fast:
                    price = float(fast["last_price"])
        except Exception:
            pass

        if price is None:
            hist = tk.history(period="1d", auto_adjust=True)
            if isinstance(hist, pd.DataFrame) and not hist.empty and "Close" in hist.columns:
                price = float(hist["Close"].iloc[-1])

        # Analyst target
        info = {}
        try:
            info = tk.info
        except Exception as e:
            print(f"[Yahoo] INFO fetch failed for {ticker}: {e}")

        if info:
            t = info.get("targetMeanPrice") or info.get("targetMedianPrice")
            if t is not None:
                target = float(t)

        if price is not None and target is not None and target != 0:
            delta_pct = (price / target - 1.0) * 100.0

    except Exception as e:
        print(f"[Yahoo] ERROR price/target for {ticker}: {e}")
        return None, None, None

    return price, target, delta_pct


def is_near_ath(ticker: str, tolerance: float = 0.03) -> Optional[bool]:
    """
    Check if current price is within 'tolerance' of 5-year high.
    Returns True/False/None if data unavailable.
    """
    try:
        hist = yf.download(
            ticker,
            period="5y",
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=False,
        )
    except Exception as e:
        print(f"[Yahoo] ERROR ATH download for {ticker}: {e}")
        return None

    if not isinstance(hist, pd.DataFrame) or hist.empty or "Close" not in hist.columns:
        return None

    close = hist["Close"].dropna()
    if close.empty:
        return None

    last = float(close.iloc[-1])
    max_close = float(close.max())
    if max_close <= 0:
        return None

    return last >= max_close * (1.0 - tolerance)

# ==========================
# MOMENTUM SCORE & TREND ANALYSIS
# ==========================

def compute_momentum_score(r10: float, r30: float, r90: float) -> float:
    """
    Weighted momentum score in percentage.
    More weight on recent performance (10d, 30d).
    """
    return (0.5 * r10 + 0.3 * r30 + 0.2 * r90) * 100.0


def bg_color_for_score(score: Optional[float]) -> str:
    """
    HTML background color based on momentum score.
    """
    if score is None:
        return "#ffffff"

    if score <= -15.0:
        return "#f8d7da"  # strong red
    if score <= -5.0:
        return "#fdecea"  # light red
    if score < 5.0:
        return "#f8f9fa"  # very light grey
    if score < 15.0:
        return "#e8f8f5"  # light green
    return "#d4efdf"      # stronger green


def classify_trend(r10: Optional[float], r30: Optional[float], r90: Optional[float]) -> str:
    """
    Rough classification of 10d/30d/90d structure.
    """
    if None in (r10, r30, r90):
        return "unknown"

    if r10 > 0 and r30 > 0 and r90 > 0:
        return "uptrend"
    if r10 < 0 and r30 < 0 and r90 < 0:
        return "downtrend"
    if r10 < 0 and r30 < 0 and r90 > 0:
        return "pullback_after_run"
    if r10 < 0 and r30 > 0 and r90 > 0:
        return "dip"
    if r10 > 0 and r30 < 0 and r90 < 0:
        return "bounce_in_downtrend"
    return "sideways"


def compute_dca_decision(
    ticker: str,
    r10: Optional[float],
    r30: Optional[float],
    r90: Optional[float],
    is_core_target: bool,
    near_ath_flag: Optional[bool],
    price_target_delta_pct: Optional[float],
) -> Tuple[str, float, Optional[float]]:
    """
    Compute final DCA action and factor for a given ticker,
    combining:
      - trend structure (10d/30d/90d)
      - momentum score
      - near ATH
      - price vs analyst target
      - core / non-core status
    Returns (action_label, dca_factor, momentum_score).
    """

    trend = classify_trend(r10, r30, r90)

    if None not in (r10, r30, r90):
        score = compute_momentum_score(r10, r30, r90)
    else:
        score = None

    # --- Base decision from trend ---
    if trend in ("uptrend", "dip", "pullback_after_run"):
        action = "ACCELERATE"
        factor = 1.15
    elif trend == "downtrend":
        action = "SLOW DOWN"
        factor = 0.85
    else:  # sideways / unknown / bounce_in_downtrend
        action = "MAINTAIN"
        factor = 1.00

    # --- Adjust with momentum intensity ---
    if score is not None:
        if score >= 20 and trend in ("uptrend", "dip", "pullback_after_run"):
            action = "ACCELERATE STRONGLY"
            factor = 1.30
        elif score <= -20 and trend == "downtrend":
            action = "SLOW DOWN"
            factor = 0.80

    # --- ATH adjustment (never 'strongly' on an ATH) ---
    if near_ath_flag:
        if action == "ACCELERATE STRONGLY":
            action = "ACCELERATE"
            factor = 1.15

    # --- Price vs analyst target adjustment ---
    if price_target_delta_pct is not None:
        if price_target_delta_pct > 15.0:
            # price well above target -> more prudence
            if action == "ACCELERATE STRONGLY":
                action = "ACCELERATE"
                factor = 1.15
            elif action == "ACCELERATE":
                action = "MAINTAIN"
                factor = 1.00
        elif price_target_delta_pct < -15.0 and trend != "downtrend":
            # price well below target in non-downtrend -> plus agressif
            if action == "MAINTAIN":
                action = "ACCELERATE"
                factor = 1.15
            elif action == "ACCELERATE":
                action = "ACCELERATE STRONGLY"
                factor = 1.30

    # --- Core portfolio protection (never slow down trop fort) ---
    if is_core_target:
        if action == "SLOW DOWN":
            action = "MAINTAIN"
            factor = max(factor, 0.90)

    # Normalize factor by action (pour éviter incohérences)
    if action == "ACCELERATE STRONGLY":
        factor = 1.30
    elif action == "ACCELERATE":
        factor = 1.15
    elif action == "MAINTAIN":
        factor = max(1.00, factor) if is_core_target else 1.00
    elif action == "SLOW DOWN":
        factor = 0.85

    return action, factor, score

# ==========================
# SIGNAL CLASSIFICATION (internal)
# ==========================

def classify_signal(returns: Dict[str, float]) -> str:
    """
    Classify ticker based on average of 10d, 30d, 90d returns.
    Used internally for allocation / email trigger, not displayed.
    """
    avg = (returns["R10"] + returns["R30"] + returns["R90"]) / 3.0
    if avg <= PANIC_THRESHOLD:
        return "PANIC_BUY"
    elif avg >= EUPHORIA_THRESHOLD:
        return "EUPHORIA"
    return "NORMAL"

# ==========================
# ALLOCATION LOGIC (internal)
# ==========================

def compute_allocation(
    signals: Dict[str, str],
    budget: float,
) -> Tuple[Dict[str, float], float]:
    """
    Weighted allocation based on signal strength, only for core AI tickers.

    CORE_AI_TICKERS are used for allocation:
        PANIC_BUY  -> overweight
        NORMAL     -> neutral
        EUPHORIA   -> underweight
    """
    allocation: Dict[str, float] = {t: 0.0 for t in signals.keys()}
    weights: Dict[str, float] = {}

    for ticker, signal in signals.items():
        if ticker not in CORE_AI_TICKERS:
            continue
        weight = WEIGHTS.get(signal, 0.0)
        if weight > 0:
            weights[ticker] = weight

    total_weight = sum(weights.values())

    if total_weight == 0:
        return allocation, budget

    for ticker, weight in weights.items():
        allocation[ticker] = round(budget * weight / total_weight, 2)

    cash_remainder = round(budget - sum(allocation.values()), 2)

    if abs(cash_remainder) >= 0.01:
        first = next(iter(weights.keys()))
        allocation[first] = round(allocation[first] + cash_remainder, 2)
        cash_remainder = 0.0

    return allocation, cash_remainder

# ==========================
# EMAIL CONTENT (TEXT + HTML)
# ==========================

def build_email_content(
    results: Dict[str, dict],
    allocation: Dict[str, float],
    cash_remainder: float,
    had_error: bool,
) -> Tuple[str, str]:
    """
    Build plain-text and HTML email bodies.

    Sections:
    - Summary
    - Multi-horizon performance (10d / 30d / 90d / Momentum)
    - DCA pacing recommendation (action finale + facteur)
    - Explanation block
    - Market vs analyst target
    - Target portfolio allocation
    """

    # --------- Summary ---------
    signals_list = [info.get("signal") for info in results.values() if info.get("signal")]
    has_panic = any(s == "PANIC_BUY" for s in signals_list)
    has_euphoria = any(s == "EUPHORIA" for s in signals_list)

    if has_panic and has_euphoria:
        summary_text = (
            "Strong divergences detected: some tickers show strong downside moves, "
            "while others show strong upside moves across recent horizons. "
            "Rebalancing may be appropriate."
        )
    elif has_panic:
        summary_text = (
            "Some tickers show strong downside moves over recent horizons. "
            "This may represent opportunities to buy the dip gradually."
        )
    elif has_euphoria:
        summary_text = (
            "Some tickers show strong upside moves over recent horizons. "
            "Consider moderating the pace of new contributions on the hottest names."
        )
    else:
        summary_text = (
            "No extreme moves detected across 10/30/90-day horizons. "
            "The portfolio looks relatively balanced from a momentum perspective."
        )

    if had_error:
        summary_text += " Data errors occurred for some tickers. Please review values carefully."

    summary_html = summary_text

    # --------- Build rows with returns & helper data ---------
    rows = []
    for ticker, info in results.items():
        rets = info.get("returns")
        if isinstance(rets, dict):
            r10 = rets.get("R10")
            r30 = rets.get("R30")
            r90 = rets.get("R90")
            if None not in (r10, r30, r90):
                avg = (r10 + r30 + r90) / 3.0
            else:
                avg = None
        else:
            r10 = r30 = r90 = None
            avg = None

        rows.append((ticker, r10, r30, r90, info.get("signal"), avg))

    def sort_key(row):
        ticker, r10, r30, r90, signal, avg = row
        if avg is None:
            return float("-1e9")
        return avg

    rows_sorted = sorted(rows, key=sort_key, reverse=True)

    def fmt(p: Optional[float]) -> str:
        return "n/a" if p is None else f"{p*100: .2f}%"

    # --------- Text version ---------
    text_lines: List[str] = []
    text_lines.append(summary_text)
    text_lines.append("")
    text_lines.append("Multi-horizon performance (sorted by average 10d/30d/90d return):")
    text_lines.append("Ticker |      10d |      30d |      90d | Momentum | DCA action | Factor")

    dca_info_for_table: List[Tuple[str, Optional[float], str, float]] = []

    for ticker, r10, r30, r90, signal, avg in rows_sorted:
        info = results.get(ticker, {})
        is_core_target = ticker in PORTFOLIO_TARGETS
        near_ath_flag = info.get("is_near_ath")
        price_delta = info.get("price_target_delta_pct")

        action, factor, score = compute_dca_decision(
            ticker,
            r10,
            r30,
            r90,
            is_core_target=is_core_target,
            near_ath_flag=near_ath_flag,
            price_target_delta_pct=price_delta,
        )

        score_str = "n/a" if score is None else f"{score:.1f}%"
        text_lines.append(
            f"{ticker:6s} | {fmt(r10):>8s} | {fmt(r30):>8s} | {fmt(r90):>8s} | "
            f"{score_str:>8s} | {action:18s} | x{factor:.2f}"
        )

        dca_info_for_table.append((ticker, score, action, factor))

    # --------- HTML table: multi-horizon + momentum ---------
    html_rows = []
    for ticker, r10, r30, r90, signal, avg in rows_sorted:
        info = results.get(ticker, {})
        is_core_target = ticker in PORTFOLIO_TARGETS
        near_ath_flag = info.get("is_near_ath")
        price_delta = info.get("price_target_delta_pct")

        action, factor, score = compute_dca_decision(
            ticker,
            r10,
            r30,
            r90,
            is_core_target=is_core_target,
            near_ath_flag=near_ath_flag,
            price_target_delta_pct=price_delta,
        )

        r10s, r30s, r90s = fmt(r10), fmt(r30), fmt(r90)
        score_str = "n/a" if score is None else f"{score:.1f}%"
        row_bg = bg_color_for_score(score)

        html_rows.append(
            f"<tr style='background-color:{row_bg};'>"
            f"<td style='padding:4px 8px; border:1px solid #ddd;'>{ticker}</td>"
            f"<td style='padding:4px 8px; border:1px solid #ddd; text-align:right;'>{r10s}</td>"
            f"<td style='padding:4px 8px; border:1px solid #ddd; text-align:right;'>{r30s}</td>"
            f"<td style='padding:4px 8px; border:1px solid #ddd; text-align:right;'>{r90s}</td>"
            f"<td style='padding:4px 8px; border:1px solid #ddd; text-align:right;'>{score_str}</td>"
            "</tr>"
        )

    html_table = (
        "<table style='border-collapse:collapse; font-family:Arial, sans-serif; font-size:13px;'>"
        "<thead>"
        "<tr>"
        "<th style='padding:4px 8px; border:1px solid #ddd; text-align:left;'>Ticker</th>"
        "<th style='padding:4px 8px; border:1px solid #ddd; text-align:right;'>10d</th>"
        "<th style='padding:4px 8px; border:1px solid #ddd; text-align:right;'>30d</th>"
        "<th style='padding:4px 8px; border:1px solid #ddd; text-align:right;'>90d</th>"
        "<th style='padding:4px 8px; border:1px solid #ddd; text-align:right;'>Momentum</th>"
        "</tr>"
        "</thead>"
        "<tbody>"
        + "".join(html_rows) +
        "</tbody>"
        "</table>"
    )

    # --------- DCA pacing recommendation table ---------
    dca_rows_html: List[str] = []
    for ticker, score, action, mult in dca_info_for_table:
        score_str = "n/a" if score is None else f"{score:.1f}%"
        dca_rows_html.append(
            "<tr>"
            f"<td style='padding:4px 8px; border:1px solid #ddd;'>{ticker}</td>"
            f"<td style='padding:4px 8px; border:1px solid #ddd; text-align:right;'>{score_str}</td>"
            f"<td style='padding:4px 8px; border:1px solid #ddd;'>{action}</td>"
            f"<td style='padding:4px 8px; border:1px solid #ddd; text-align:right;'>×{mult:.2f}</td>"
            "</tr>"
        )

    dca_table = (
        "<h3 style='margin-top:16px;'>DCA pacing recommendation</h3>"
        "<table style='border-collapse:collapse; font-family:Arial, sans-serif; font-size:13px;'>"
        "<thead>"
        "<tr>"
        "<th style='padding:4px 8px; border:1px solid #ddd; text-align:left;'>Ticker</th>"
        "<th style='padding:4px 8px; border:1px solid #ddd; text-align:right;'>Momentum</th>"
        "<th style='padding:4px 8px; border:1px solid #ddd; text-align:left;'>Action</th>"
        "<th style='padding:4px 8px; border:1px solid #ddd; text-align:right;'>DCA factor</th>"
        "</tr>"
        "</thead>"
        "<tbody>"
        + "".join(dca_rows_html) +
        "</tbody>"
        "</table>"
    )

    # --------- Explanation of DCA logic ---------
    dca_explanation = """
<h4 style="margin-top:12px;">How to read these DCA recommendations</h4>
<ul style="margin-top:4px;">
  <li><strong>Momentum</strong> is a weighted score: 50% × 10d + 30% × 30d + 20% × 90d performance.</li>
  <li><strong>ACCELERATE / ACCELERATE STRONGLY</strong> means the trend is constructive (uptrend or healthy dip) and momentum is positive.</li>
  <li><strong>MAINTAIN</strong> means the signal is mixed or neutral: you continue your normal DCA without change.</li>
  <li><strong>SLOW DOWN</strong> is used only when the structure (10d/30d/90d) looks like a real downtrend, and mostly on non-core positions.</li>
  <li>If the price is near a <strong>multi-year high (ATH)</strong>, the model never recommends "ACCELERATE STRONGLY" to avoid chasing extreme breakouts.</li>
  <li>If the market price is far <strong>above</strong> the average analyst target (&gt; +15%), the DCA is nudged one step more cautious. If it is far <strong>below</strong> (&lt; -15%) in a non-bearish trend, the DCA can be nudged more aggressive.</li>
  <li><strong>Core portfolio names</strong> (those in your long-term target allocation) have a floor: the model does not slow them down too aggressively, to avoid missing the next "NVDA-like" cycle.</li>
  <li>The <strong>DCA factor</strong> is a simple multiplicative adjustment to your usual monthly contribution (e.g. ×1.15 = +15% this month).</li>
</ul>
"""

    # --------- Price vs analyst target table ---------
    price_rows_html: List[str] = []
    for ticker, info in results.items():
        price = info.get("price")
        target = info.get("target")
        delta = info.get("price_target_delta_pct")

        if price is None and target is None:
            continue

        if price is None:
            price_str = "n/a"
        else:
            price_str = f"{price:,.2f}"

        if target is None:
            target_str = "n/a"
        else:
            target_str = f"{target:,.2f}"

        if delta is None:
            delta_str = "n/a"
            row_bg = "#ffffff"
            status = "n/a"
        else:
            delta_str = f"{delta:+.1f}%"
            if delta > 15.0:
                row_bg = "#fdecea"
                status = "Above target (rich)"
            elif delta < -15.0:
                row_bg = "#e8f8f5"
                status = "Below target (cheap)"
            else:
                row_bg = "#f8f9fa"
                status = "Near target"

        price_rows_html.append(
            f"<tr style='background-color:{row_bg};'>"
            f"<td style='padding:4px 8px; border:1px solid #ddd;'>{ticker}</td>"
            f"<td style='padding:4px 8px; border:1px solid #ddd; text-align:right;'>{price_str}</td>"
            f"<td style='padding:4px 8px; border:1px solid #ddd; text-align:right;'>{target_str}</td>"
            f"<td style='padding:4px 8px; border:1px solid #ddd; text-align:right;'>{delta_str}</td>"
            f"<td style='padding:4px 8px; border:1px solid #ddd;'>{status}</td>"
            "</tr>"
        )

    if price_rows_html:
        price_target_table = (
            "<h3 style='margin-top:16px;'>Market price vs analyst target</h3>"
            "<table style='border-collapse:collapse; font-family:Arial, sans-serif; font-size:13px;'>"
            "<thead>"
            "<tr>"
            "<th style='padding:4px 8px; border:1px solid #ddd; text-align:left;'>Ticker</th>"
            "<th style='padding:4px 8px; border:1px solid #ddd; text-align:right;'>Price</th>"
            "<th style='padding:4px 8px; border:1px solid #ddd; text-align:right;'>Analyst target</th>"
            "<th style='padding:4px 8px; border:1px solid #ddd; text-align:right;'>Delta</th>"
            "<th style='padding:4px 8px; border:1px solid #ddd; text-align:left;'>Status</th>"
            "</tr>"
            "</thead>"
            "<tbody>"
            + "".join(price_rows_html) +
            "</tbody>"
            "</table>"
        )
    else:
        price_target_table = ""

    # --------- Target portfolio allocation table ---------
    pt_rows_html: List[str] = []
    for t, w in PORTFOLIO_TARGETS.items():
        pt_rows_html.append(
            "<tr>"
            f"<td style='padding:4px 8px; border:1px solid #ddd;'>{t}</td>"
            f"<td style='padding:4px 8px; border:1px solid #ddd; text-align:right;'>{w}%</td>"
            "</tr>"
        )

    portfolio_table = (
        "<h3 style='margin-top:16px;'>Target portfolio allocation</h3>"
        "<table style='border-collapse:collapse; font-family:Arial, sans-serif; font-size:13px;'>"
        "<thead>"
        "<tr>"
        "<th style='padding:4px 8px; border:1px solid #ddd; text-align:left;'>Ticker</th>"
        "<th style='padding:4px 8px; border:1px solid #ddd; text-align:right;'>Target weight</th>"
        "</tr>"
        "</thead>"
        "<tbody>"
        + "".join(pt_rows_html) +
        "</tbody>"
        "</table>"
    )

    text_body = "\n".join(text_lines)

    html_body = f"""\
<html>
  <body style="font-family: Arial, sans-serif; font-size: 13px; color: #2c3e50;">
    <p>{summary_html}</p>
    <h3 style="margin-top:16px;">Multi-horizon performance and signals</h3>
    {html_table}
    {dca_table}
    {dca_explanation}
    {price_target_table}
    {portfolio_table}
  </body>
</html>
"""
    return text_body, html_body

# ==========================
# EMAIL SEND
# ==========================

def send_email(subject: str, text_body: str, html_body: str) -> None:
    """
    Send a multipart email (plain text + HTML) using SMTP settings from environment variables.
    """
    host = os.getenv("SMTP_HOST")
    port = os.getenv("SMTP_PORT")
    user = os.getenv("SMTP_USER")
    password = os.getenv("SMTP_PASSWORD")
    email_from = os.getenv("EMAIL_FROM")
    email_to = os.getenv("EMAIL_TO") or user

    if not all([host, port, user, password, email_from, email_to]):
        print("[Email] Missing SMTP configuration — cannot send email.")
        print(f"host={host}, port={port}, user={user}, from={email_from}, to={email_to}")
        return

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = email_from
    msg["To"] = email_to

    part_text = MIMEText(text_body, "plain", "utf-8")
    part_html = MIMEText(html_body, "html", "utf-8")

    msg.attach(part_text)
    msg.attach(part_html)

    try:
        with smtplib.SMTP(host, int(port), timeout=20) as server:
            server.starttls()
            server.login(user, password)
            server.send_message(msg)
        print(f"[Email] Report sent successfully to {email_to}")
    except Exception as e:
        print(f"[Email] ERROR while sending email: {e}")

# ==========================
# GENERATE HTML FILE 
# ==========================

def save_html_report(html_body: str, filename: str = "ai_signals_report.html") -> None:
    """Save the HTML report to a file."""
    with open(filename, "w", encoding="utf-8") as f:
        f.write(html_body)
    print(f"[HTML] Report written to {filename}")

# ==========================
# EMAIL MODE DECISION
# ==========================

def should_send_email(
    results: Dict[str, dict],
    had_error: bool,
    mode: str,
) -> bool:
    """
    Decide whether an email should be sent based on mode and signals.
    """
    mode = (mode or "only_on_action").lower().strip()

    if mode == "never":
        return False
    if mode == "always":
        return True

    if had_error:
        return True

    for info in results.values():
        signal = info.get("signal")
        source = info.get("source")
        returns = info.get("returns")

        if signal in ("PANIC_BUY", "EUPHORIA"):
            return True
        if source in ("MISMATCH", "NONE"):
            return True
        if returns is None:
            return True

    return False

# ==========================
# MAIN EXECUTION
# ==========================

def run_signals(budget: float, email_mode: str, dry_run: bool = False) -> int:
    print(f"=== AI Robust Signals {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} ===\n")

    results: Dict[str, dict] = {}
    signals: Dict[str, str] = {}
    had_error = False

    def fmt(p: Optional[float]) -> str:
        return "n/a" if p is None else f"{p*100: .2f}%"

    for ticker in TICKERS:
        try:
            returns, source, sec_returns = get_returns_robust(ticker)
        except Exception as e:
            had_error = True
            print(f"[ERROR] Unexpected error while processing {ticker}: {e}")
            results[ticker] = {
                "returns": None,
                "source": "ERROR",
                "secondary_returns": None,
                "signal": "ERROR",
                "price": None,
                "target": None,
                "price_target_delta_pct": None,
                "is_near_ath": None,
            }
            continue

        # Price / target / ATH (best effort)
        price, target, delta_pct = get_price_and_target(ticker)
        ath_flag = is_near_ath(ticker)

        if returns is None:
            print(f"{ticker:8s} | NO DATA")
            results[ticker] = {
                "returns": None,
                "source": source,
                "secondary_returns": sec_returns,
                "signal": "NO_DATA",
                "price": price,
                "target": target,
                "price_target_delta_pct": delta_pct,
                "is_near_ath": ath_flag,
            }
            continue

        signal = classify_signal(returns)
        signals[ticker] = signal

        r10, r30, r90 = returns["R10"], returns["R30"], returns["R90"]
        pct_str = f"10d {fmt(r10)} 30d {fmt(r30)} 90d {fmt(r90)}"
        extra = ""
        if source == "MISMATCH" and sec_returns is not None:
            extra = " (MISMATCH vs TwelveData)"

        print(
            f"{ticker:8s} | {pct_str} | Source: {source:9s} | Signal: {signal}{extra}"
        )

        results[ticker] = {
            "returns": returns,
            "source": source,
            "secondary_returns": sec_returns,
            "signal": signal,
            "price": price,
            "target": target,
            "price_target_delta_pct": delta_pct,
            "is_near_ath": ath_flag,
        }

    print("\n--- Allocation suggestion (core AI only, based on internal signals) ---")
    allocation, cash_remainder = compute_allocation(signals, budget)

    core_alloc_items = [
        (t, allocation.get(t, 0.0))
        for t in CORE_AI_TICKERS
        if allocation.get(t, 0.0) > 0
    ]

    if not core_alloc_items:
        print(f"No valid allocation for core AI tickers. Keep {budget:.2f} USD as cash.")
    else:
        for t, amount in core_alloc_items:
            print(f"{t:8s} -> {amount:8.2f} USD")
        if cash_remainder > 0:
            print(f"Cash remainder: {cash_remainder:.2f} USD")

    # Build email / HTML content
    text_body, html_body = build_email_content(results, allocation, cash_remainder, had_error)

    # Always save HTML locally
    save_html_report(html_body)

    email_mode_env = os.getenv("EMAIL_MODE", email_mode or "only_on_action")

    if should_send_email(results, had_error, email_mode_env):
        subject = "AI Signals Report"
        if dry_run:
            print("\n[DRY-RUN] Email would be sent with subject:", subject)
        else:
            print("\n[INFO] Sending email report...")
            send_email(subject, text_body, html_body)
    else:
        print("\n[INFO] No email sent (mode =", email_mode_env, ")")

    return 1 if had_error else 0

# ==========================
# CLI
# ==========================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AI signals and allocation helper (Yahoo + TwelveData, GitHub Actions friendly)"
    )
    parser.add_argument(
        "--budget",
        type=float,
        default=DEFAULT_BUDGET,
        help=f"Monthly budget in USD (default: {DEFAULT_BUDGET})",
    )
    parser.add_argument(
        "--email-mode",
        type=str,
        default="only_on_action",
        choices=["only_on_action", "always", "never"],
        help="Email mode: only_on_action (default), always, or never. "
             "This can also be controlled via EMAIL_MODE env var.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do everything except actually sending emails.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    exit_code = run_signals(
        budget=args.budget,
        email_mode=args.email_mode,
        dry_run=args.dry_run,
    )
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
