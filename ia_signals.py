from __future__ import annotations

import argparse
import os
import smtplib
from datetime import datetime, timezone
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from io import StringIO
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
    "AMD",
    "COST",
    "QQQ",
]

# Core AI tickers used for allocation and IBKR suggestions
CORE_AI_TICKERS = ["NVDA", "MSFT", "GOOGL", "META", "AMZN", "ASML", "AMD"]

# Tickers shown only for information (no effect on allocation)
INFO_ONLY_TICKERS = ["COST", "QQQ"]

# 5-trading-day performance thresholds
PANIC_THRESHOLD = -0.08    # -8% over last 5 sessions
EUPHORIA_THRESHOLD = 0.10  # +10% over last 5 sessions

# Data consistency tolerance between Yahoo and Twelve Data
MISMATCH_TOLERANCE = 0.01  # 1 percentage point

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
    # "AMD": 0.0,  # define when you start a recurring amount for AMD
    # "QQQ": 0.0,
}

# ==========================
# DATA SOURCES
# ==========================

def download_yahoo_close(ticker: str, lookback_days: int = 10) -> Optional[pd.Series]:
    """Robust Yahoo Finance download. Returns Series or None."""
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
# 5-DAY PERFORMANCE
# ==========================

def compute_5d_change(series: pd.Series) -> float:
    """Compute 5-trading-day change: (last / close_5_sessions_ago - 1)."""
    if not isinstance(series, pd.Series):
        raise TypeError(f"Input is not Series: {type(series)}")

    series = series.dropna()

    if len(series) < 6:
        raise ValueError("Not enough data points to compute 5-day change")

    last = float(series.iloc[-1])
    prev_5 = float(series.iloc[-6])

    return (last / prev_5) - 1.0


def get_5d_change_robust(ticker: str) -> Tuple[Optional[float], str, Optional[float]]:
    """Return a robust 5-day performance estimate using Yahoo + Twelve Data."""
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
# SIGNAL CLASSIFICATION
# ==========================

def classify_signal(change_5d: float) -> str:
    if change_5d <= PANIC_THRESHOLD:
        return "PANIC_BUY"
    if change_5d >= EUPHORIA_THRESHOLD:
        return "EUPHORIA"
    return "NORMAL"

# ==========================
# ALLOCATION LOGIC
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

    INFO_ONLY_TICKERS are ignored for allocation.
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
    - 5-day performance and signals (all tickers, sorted by 5d change)
    - Recommended allocation for AI leaders
    - IBKR adjustments for this month
    """

    # Summary
    signals_list = [info.get("signal") for info in results.values() if info.get("signal")]
    has_panic = any(s == "PANIC_BUY" for s in signals_list)
    has_euphoria = any(s == "EUPHORIA" for s in signals_list)

    if has_panic and has_euphoria:
        summary_text = "Some tickers are in PANIC and some are in EUPHORIA. Rebalancing is recommended."
    elif has_panic:
        summary_text = "Some tickers are in PANIC. Additional investment is recommended."
    elif has_euphoria:
        summary_text = "Some tickers are in EUPHORIA. Reduced exposure is recommended."
    else:
        summary_text = "All signals are normal. No action required this month."

    if had_error:
        summary_text += " Data errors occurred for some tickers. Please review values carefully."

    summary_html = summary_text
    summary_html = summary_html.replace(
        "PANIC",
        '<span style="font-weight:bold; color:#c0392b;">PANIC</span>'
    )
    summary_html = summary_html.replace(
        "EUPHORIA",
        '<span style="font-weight:bold; color:#e67e22;">EUPHORIA</span>'
    )

    # 5-day performance table (sorted)
    rows = []
    for ticker, info in results.items():
        change = info.get("change_5d")
        signal = info.get("signal")
        rows.append((ticker, change, signal))

    def sort_key(row):
        ticker, change, signal = row
        if change is None:
            return float("-inf")
        return change

    rows_sorted = sorted(rows, key=sort_key, reverse=True)

    text_lines: List[str] = []
    text_lines.append(summary_text)
    text_lines.append("")
    text_lines.append("5-day performance and signals (sorted by 5d change):")
    text_lines.append("Ticker | 5d Change | Signal")

    for ticker, change, signal in rows_sorted:
        if change is None:
            change_str = "n/a"
        else:
            change_str = f"{change * 100: .2f}%"
        text_lines.append(f"{ticker:6s} | {change_str:>8s} | {signal or 'n/a'}")

    html_rows = []
    for ticker, change, signal in rows_sorted:
        if change is None:
            change_str = "n/a"
        else:
            change_str = f"{change * 100: .2f}%"

        if signal == "PANIC_BUY":
            signal_label = "PANIC"
            color = "#c0392b"
        elif signal == "EUPHORIA":
            signal_label = "EUPHORIA"
            color = "#e67e22"
        elif signal == "NORMAL":
            signal_label = "NORMAL"
            color = "#2c3e50"
        else:
            signal_label = signal or "n/a"
            color = "#7f8c8d"

        # ✅ version sans bug de guillemets
        signal_html = f"<span style='font-weight:bold; color:{color};'>{signal_label}</span>"

        html_rows.append(
            "<tr>"
            f"<td style='padding:4px 8px; border:1px solid #ddd;'>{ticker}</td>"
            f"<td style='padding:4px 8px; border:1px solid #ddd; text-align:right;'>{change_str}</td>"
            f"<td style='padding:4px 8px; border:1px solid #ddd;'>{signal_html}</td>"
            "</tr>"
        )


    html_table = (
        "<table style='border-collapse:collapse; font-family:Arial, sans-serif; font-size:13px;'>"
        "<thead>"
        "<tr>"
        "<th style='padding:4px 8px; border:1px solid #ddd; text-align:left;'>Ticker</th>"
        "<th style='padding:4px 8px; border:1px solid #ddd; text-align:right;'>5d Change</th>"
        "<th style='padding:4px 8px; border:1px solid #ddd; text-align:left;'>Signal</th>"
        "</tr>"
        "</thead>"
        "<tbody>"
        + "".join(html_rows) +
        "</tbody>"
        "</table>"
    )

    # Allocation for core AI leaders
    core_alloc_items = [
        (t, allocation.get(t, 0.0))
        for t in CORE_AI_TICKERS
        if allocation.get(t, 0.0) > 0
    ]
    total_core_alloc = sum(a for _, a in core_alloc_items)

    text_lines.append("")
    text_lines.append(
        f"Recommended monthly allocation for AI leaders ({total_core_alloc:.0f} USD):"
    )
    for ticker, amount in core_alloc_items:
        text_lines.append(f"{ticker:6s} {amount:.2f}")

    html_alloc_rows = []
    for ticker, amount in core_alloc_items:
        html_alloc_rows.append(
            "<tr>"
            f"<td style='padding:4px 8px; border:1px solid #ddd;'>{ticker}</td>"
            f"<td style='padding:4px 8px; border:1px solid #ddd; text-align:right;'>{amount:.2f}</td>"
            "</tr>"
        )

    html_alloc_table = (
        "<table style='border-collapse:collapse; font-family:Arial, sans-serif; font-size:13px; margin-top:8px;'>"
        "<thead>"
        "<tr>"
        "<th style='padding:4px 8px; border:1px solid #ddd; text-align:left;'>Ticker</th>"
        "<th style='padding:4px 8px; border:1px solid #ddd; text-align:right;'>Monthly amount (USD)</th>"
        "</tr>"
        "</thead>"
        "<tbody>"
        + "".join(html_alloc_rows) +
        "</tbody>"
        "</table>"
    )

    # IBKR adjustments
    text_lines.append("")
    text_lines.append("IBKR adjustments for this month:")

    ibkr_lines: List[str] = []
    ibkr_html_rows: List[str] = []

    for ticker, amount in core_alloc_items:
        old = IBKR_REFERENCE_MONTHLY.get(ticker)
        if old is None:
            continue
        if abs(old - amount) < 1.0:
            continue
        ibkr_lines.append(
            f"{ticker}:\n"
            f"Change monthly investment from {old:.0f} to {amount:.0f}\n"
        )
        ibkr_html_rows.append(
            "<p style='margin:4px 0;'>"
            f"<strong>{ticker}:</strong><br>"
            f"Change monthly investment from {old:.0f} to {amount:.0f}"
            "</p>"
        )

    if not ibkr_lines:
        text_lines.append("No changes required. Keep your current automatic investments.")
        ibkr_html_block = "<p>No changes required. Keep your current automatic investments.</p>"
    else:
        text_lines.extend(ibkr_lines)
        ibkr_html_block = "".join(ibkr_html_rows)

    text_body = "\n".join(text_lines)

    html_body = f"""\
<html>
  <body style="font-family: Arial, sans-serif; font-size: 13px; color: #2c3e50;">
    <p>{summary_html}</p>
    <h3 style="margin-top:16px;">5-day performance and signals</h3>
    {html_table}
    <h3 style="margin-top:16px;">Recommended monthly allocation for AI leaders</h3>
    {html_alloc_table}
    <h3 style="margin-top:16px;">IBKR adjustments for this month</h3>
    {ibkr_html_block}
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

    Required environment variables:
        SMTP_HOST
        SMTP_PORT
        SMTP_USER
        SMTP_PASSWORD
        EMAIL_FROM

    Optional:
        EMAIL_TO (if missing -> defaults to SMTP_USER)
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
# EMAIL MODE DECISION
# ==========================

def should_send_email(
    results: Dict[str, dict],
    had_error: bool,
    mode: str,
) -> bool:
    """
    Decide whether an email should be sent based on mode and signals.

    mode:
        - "always"
        - "only_on_action" (default)
        - "never"
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
        change_5d = info.get("change_5d")

        if signal in ("PANIC_BUY", "EUPHORIA"):
            return True
        if source in ("MISMATCH", "NONE"):
            return True
        if change_5d is None:
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

    for ticker in TICKERS:
        try:
            change_5d, source, sec_change = get_5d_change_robust(ticker)
        except Exception as e:
            had_error = True
            print(f"[ERROR] Unexpected error while processing {ticker}: {e}")
            results[ticker] = {
                "change_5d": None,
                "source": "ERROR",
                "secondary_change_5d": None,
                "signal": "ERROR",
            }
            continue

        if change_5d is None:
            print(f"{ticker:8s} | NO DATA")
            results[ticker] = {
                "change_5d": None,
                "source": source,
                "secondary_change_5d": sec_change,
                "signal": "NO_DATA",
            }
            continue

        signal = classify_signal(change_5d)
        signals[ticker] = signal

        pct_str = f"{change_5d * 100: .2f}%"
        extra = ""
        if source == "MISMATCH" and sec_change is not None:
            extra = f" (TwelveData: {sec_change*100: .2f}%)"

        print(
            f"{ticker:8s} | 5d: {pct_str:>8} | "
            f"Source: {source:9s} | Signal: {signal}{extra}"
        )

        results[ticker] = {
            "change_5d": change_5d,
            "source": source,
            "secondary_change_5d": sec_change,
            "signal": signal,
        }

    print("\n--- Allocation suggestion (core AI only) ---")
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

    # Email content
    text_body, html_body = build_email_content(results, allocation, cash_remainder, had_error)
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
