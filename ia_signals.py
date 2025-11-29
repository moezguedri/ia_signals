from __future__ import annotations

import argparse
import os
import smtplib
import urllib3
from datetime import datetime, timezone
from email.mime.text import MIMEText
from io import StringIO
from typing import Optional, Dict, List, Tuple

import pandas as pd
import requests
import yfinance as yf
import certifi


# ==========================
# SSL / CERTIFICATE SETUP
# ==========================

# Disable noisy warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Force Python / Requests / yfinance to use certifi certificates
CA_BUNDLE = certifi.where()
os.environ["SSL_CERT_FILE"] = CA_BUNDLE
os.environ["REQUESTS_CA_BUNDLE"] = CA_BUNDLE
os.environ["CURL_CA_BUNDLE"] = CA_BUNDLE


# ==========================
# CONFIGURATION
# ==========================

TICKERS: List[str] = [
    "NVDA",
    "MSFT",
    "GOOGL",
    "META",
    "AMZN",
    "ASML",
    "QQQ",
]

PANIC_THRESHOLD = -0.08
EUPHORIA_THRESHOLD = 0.10
MISMATCH_TOLERANCE = 0.01
DEFAULT_BUDGET = 4000.0

STOOQ_OVERRIDE: Dict[str, str] = {
    # "ASML": "ASML.US",
    # "ASML.AS": "ASML.NL",
    # "QQQ": "QQQ.US",
}


# ==========================
# DATA SOURCES
# ==========================

def download_yahoo_close(ticker: str, lookback_days: int = 10) -> Optional[pd.Series]:
    """Robust Yahoo Finance download. Always returns Series or None."""
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

    # NEW: handle both Series and DataFrame for Close
    if isinstance(close, pd.DataFrame):
        # If MultiIndex columns (e.g. (field, ticker)), try to select the right one
        if isinstance(close.columns, pd.MultiIndex):
            # try to select column where second level == ticker
            try:
                close = close.xs(ticker, axis=1, level=-1)
            except Exception:
                # fallback: take first column
                close = close.iloc[:, 0]
        else:
            # Simple DataFrame: take the first column
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
# 5 DAY CHANGE
# ==========================

def compute_5d_change(series: pd.Series) -> float:
    if not isinstance(series, pd.Series):
        raise TypeError(f"Input is not Series: {type(series)}")

    series = series.dropna()

    if len(series) < 6:
        raise ValueError("Not enough data points")

    last = float(series.iloc[-1])
    prev_5 = float(series.iloc[-6])

    return (last / prev_5) - 1.0


def get_5d_change_robust(ticker: str) -> Tuple[Optional[float], str, Optional[float]]:
    yahoo = download_yahoo_close(ticker)
    twelve = download_twelve_close(ticker)

    yahoo_change = None
    twelve_change = None

    if yahoo is not None:
        try:
            yahoo_change = compute_5d_change(yahoo)
        except Exception as e:
            print(f"[Yahoo] Compute failed for {ticker}: {e}")

    if twelve is not None:
        try:
            twelve_change = compute_5d_change(twelve)
        except Exception as e:
            print(f"[TwelveData] Compute failed for {ticker}: {e}")

    if yahoo_change is None and twelve_change is None:
        return None, "NONE", None

    if yahoo_change is not None and twelve_change is None:
        return float(yahoo_change), "YAHOO", None

    if yahoo_change is None and twelve_change is not None:
        return float(twelve_change), "TWELVEDATA", None

    diff = abs(yahoo_change - twelve_change)

    if diff > MISMATCH_TOLERANCE:
        print(
            f"[MISMATCH] {ticker}: Yahoo={yahoo_change*100:.2f}% "
            f"TwelveData={twelve_change*100:.2f}%"
        )
        return float(yahoo_change), "MISMATCH", float(twelve_change)

    return float(yahoo_change), "YAHOO", float(twelve_change)


# ==========================
# SIGNAL
# ==========================

def classify_signal(change_5d: float) -> str:
    if change_5d <= PANIC_THRESHOLD:
        return "PANIC_BUY"
    if change_5d >= EUPHORIA_THRESHOLD:
        return "EUPHORIA"
    return "NORMAL"


# ==========================
# ALLOCATION
# ==========================

WEIGHTS = {
    "PANIC_BUY": 2.0,
    "NORMAL": 1.0,
    "EUPHORIA": 0.25,
}


def compute_allocation(signals: Dict[str, str], budget: float) -> Tuple[Dict[str, float], float]:
    """
    Weighted allocation based on signal strength.

    PANIC_BUY  -> overweight
    NORMAL     -> neutral
    EUPHORIA   -> underweight (but never zero)
    """

    allocation: Dict[str, float] = {}
    weights: Dict[str, float] = {}

    # assign weights
    for ticker, signal in signals.items():
        weight = WEIGHTS.get(signal, 0)
        if weight > 0:
            weights[ticker] = weight

    total_weight = sum(weights.values())

    # edge case: nothing investable
    if total_weight == 0:
        return {t: 0.0 for t in signals}, budget

    # normalized allocation
    for ticker, weight in weights.items():
        allocation[ticker] = round(budget * weight / total_weight, 2)

    cash_remainder = round(budget - sum(allocation.values()), 2)

    # minor rounding adjustment
    if cash_remainder != 0:
        first = next(iter(allocation))
        allocation[first] += cash_remainder
        cash_remainder = 0.0

    # ensure all tickers exist in output
    for t in signals:
        allocation.setdefault(t, 0.0)

    return allocation, cash_remainder


# ==========================
# EMAIL
# ==========================

def build_report_text(results, allocation, cash, had_error) -> str:
    lines = [f"AI Robust Signals {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n"]

    for t, info in results.items():
        if info["change_5d"] is None:
            lines.append(f"{t:8s} | NO DATA")
            continue

        pct = f"{info['change_5d']*100:.2f}%"
        extra = f" (Stooq {info['secondary_change_5d']*100:.2f}%)" if info["source"] == "MISMATCH" else ""
        lines.append(f"{t:8s} | 5d {pct:>8} | {info['signal']} | {info['source']}{extra}")

    lines.append("\nAllocation suggestion:")

    for t, amt in allocation.items():
        if amt > 0:
            lines.append(f"  {t:8s} -> {amt:.2f} USD")

    if cash > 0:
        lines.append(f"  Cash remainder: {cash:.2f} USD")

    if had_error:
        lines.append("\n⚠ WARNING: Data errors occurred.")

    return "\n".join(lines)


def send_email(subject: str, body: str) -> None:
    """
    Send an email using SMTP settings from environment variables.

    Required environment variables:
        SMTP_HOST
        SMTP_PORT
        SMTP_USER
        SMTP_PASSWORD
        EMAIL_FROM

    Optional:
        EMAIL_TO (if missing -> defaults to SMTP_USER)

    Typical example (Gmail):
        SMTP_HOST = smtp.gmail.com
        SMTP_PORT = 587
        SMTP_USER = your@gmail.com
        SMTP_PASSWORD = app_password
        EMAIL_FROM = your@gmail.com
        EMAIL_TO = receiver@gmail.com   (optional)
    """

    host = os.getenv("SMTP_HOST")
    port = os.getenv("SMTP_PORT")
    user = os.getenv("SMTP_USER")
    password = os.getenv("SMTP_PASSWORD")
    email_from = os.getenv("EMAIL_FROM")

    # If EMAIL_TO is not defined, use SMTP_USER as destination (safe fallback)
    email_to = os.getenv("EMAIL_TO") or user

    if not all([host, port, user, password, email_from, email_to]):
        print("[Email] Missing SMTP configuration — cannot send email.")
        print(f"host={host}, port={port}, user={user}, from={email_from}, to={email_to}")
        return

    msg = MIMEText(body, "plain", "utf-8")
    msg["Subject"] = subject
    msg["From"] = email_from
    msg["To"] = email_to

    try:
        with smtplib.SMTP(host, int(port), timeout=20) as server:
            server.starttls()
            server.login(user, password)
            server.send_message(msg)

        print(f"[Email] Report sent successfully to {email_to}")

    except Exception as e:
        print(f"[Email] ERROR while sending email: {e}")



# ==========================
# MAIN
# ==========================

def run_signals(budget: float, email_mode: str, dry: bool = False) -> int:
    print(f"=== AI Robust Signals {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} ===\n")
    results = {}
    signals = {}
    had_error = False

    for t in TICKERS:
        try:
            change, src, other = get_5d_change_robust(t)
        except Exception as e:
            print(f"[ERROR] {t}: {e}")
            had_error = True
            results[t] = {"change_5d": None, "signal": "ERROR", "source": "ERROR", "secondary_change_5d": None}
            continue

        if change is None:
            results[t] = {"change_5d": None, "signal": "NO_DATA", "source": src, "secondary_change_5d": None}
            print(f"{t:8s} | NO DATA")
            continue

        change = float(change)
        signal = classify_signal(change)
        signals[t] = signal

        results[t] = {
            "change_5d": change,
            "signal": signal,
            "source": src,
            "secondary_change_5d": other,
        }

        line = f"{t:8s} | {change*100:>7.2f}% | {signal} | {src}"
        if src == "MISMATCH":
            line += f" (Stooq {other*100:.2f}%)"
        print(line)

    allocation, cash = compute_allocation(signals, budget)

    print("\n--- ALLOCATION ---")
    for t, amt in allocation.items():
        if amt > 0:
            print(f"{t:8s} -> {amt:.2f} USD")
    if cash > 0:
        print(f"Cash   -> {cash:.2f} USD")

    report = build_report_text(results, allocation, cash, had_error)
    mode = os.getenv("EMAIL_MODE", email_mode)

    if mode == "always" or (mode == "only_on_action" and any(s in ["PANIC_BUY", "EUPHORIA"] for s in signals.values())):
        if dry:
            print("\n[DRY-RUN] Email not sent.")
        else:
            print("\n[INFO] Sending email...")
            send_email("AI Signals Report", report)
    else:
        print("\n[INFO] No email sent.")

    return 1 if had_error else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, default=DEFAULT_BUDGET)
    ap.add_argument("--email-mode", choices=["always", "only_on_action", "never"], default="only_on_action")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    exit(run_signals(args.budget, args.email_mode, args.dry_run))


if __name__ == "__main__":
    main()
