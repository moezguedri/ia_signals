from __future__ import annotations

import argparse
import os
from datetime import datetime
from typing import Dict, Optional

import certifi
import pandas as pd
import yfinance as yf

# ==========================
# SSL / CERTIFICATES
# ==========================

# Force yfinance / requests / curl to use certifi CA bundle
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()
os.environ["CURL_CA_BUNDLE"] = certifi.where()

# ==========================
# CONFIG
# ==========================

DEFAULT_BUDGET = 4000.0

# Tickers suivis
TICKERS = ["NVDA", "MSFT", "GOOGL", "META", "AMZN", "ASML", "COST"]

# Poids actuels (approx, en % du portefeuille total)
CURRENT_WEIGHTS: Dict[str, float] = {
    "NVDA": 24.4,
    "MSFT": 18.1,
    "GOOGL": 16.3,
    "META": 12.5,
    "AMZN": 12.5,
    "ASML": 5.2,
    "COST": 11.0,
}

# Cibles long terme
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
# DATA HELPERS
# ==========================

def yahoo_close(ticker: str, days: int = 120) -> Optional[pd.Series]:
    """Télécharge les clôtures ajustées sur N jours."""
    try:
        df = yf.download(
            ticker,
            period=f"{days}d",
            interval="1d",
            auto_adjust=True,
            progress=False,
        )
    except Exception as e:
        print(f"[Yahoo] ERROR for {ticker}: {e}")
        return None

    if not isinstance(df, pd.DataFrame) or df.empty or "Close" not in df.columns:
        print(f"[Yahoo] No data for {ticker}")
        return None

    s = df["Close"].dropna()
    if len(s) < 40:
        print(f"[Yahoo] Not enough data for {ticker} (n={len(s)})")
        return None
    return s


def compute_return(series: pd.Series, days: int) -> float:
    """Return sur N séances: (last / close_N_days_ago - 1)."""
    if len(series) < days + 1:
        raise ValueError(f"Not enough data for {days}-day return")
    last = series.iloc[-1].item()
    prev = series.iloc[-(days + 1)].item()
    return (last / prev) - 1.0


def multi_returns(close: pd.Series) -> Dict[str, float]:
    """10d / 30d / 90d (décimaux, ex: 0.12 pour +12%)."""
    return {
        "R10": compute_return(close, 10),
        "R30": compute_return(close, 30),
        "R90": compute_return(close, 90),
    }


def price_and_target(ticker: str):
    """
    Prix actuel + target analystes depuis Yahoo.
    Retourne (price, target, delta_pct) avec delta_pct=(price/target-1)*100.
    """
    price = None
    target = None
    delta_pct = None
    tk = yf.Ticker(ticker)

    # Prix
    try:
        fi = getattr(tk, "fast_info", None)
        if fi and "lastPrice" in fi:
            price = float(fi["lastPrice"])
    except Exception:
        pass

    if price is None:
        try:
            hist = tk.history(period="1d", auto_adjust=True)
            if isinstance(hist, pd.DataFrame) and not hist.empty:
                price = hist["Close"].iloc[-1].item()
        except Exception:
            pass

    # Target analystes
    try:
        info = tk.info
    except Exception:
        info = {}

    if info:
        t = info.get("targetMeanPrice") or info.get("targetMedianPrice")
        if t is not None:
            target = float(t)

    if price is not None and target is not None and target != 0:
        delta_pct = (price / target - 1.0) * 100.0

    return price, target, delta_pct


def near_ath(ticker: str, tolerance: float = 0.03) -> Optional[bool]:
    """True si le prix est à moins de `tolerance` du plus haut 5 ans."""
    try:
        df = yf.download(
            ticker,
            period="5y",
            interval="1d",
            auto_adjust=True,
            progress=False,
        )
    except Exception as e:
        print(f"[Yahoo] ATH ERROR for {ticker}: {e}")
        return None

    if not isinstance(df, pd.DataFrame) or df.empty or "Close" not in df.columns:
        return None

    close = df["Close"].dropna()
    if close.empty:
        return None

    last = close.iloc[-1].item()
    high = close.max().item()
    if high <= 0:
        return None

    return last >= high * (1.0 - tolerance)

# ==========================
# MOMENTUM & DCA LOGIC
# ==========================

def momentum_score(rets: Dict[str, float]) -> float:
    """Score de momentum (%) – 50% 10d, 30% 30d, 20% 90d."""
    return (0.5 * rets["R10"] + 0.3 * rets["R30"] + 0.2 * rets["R90"]) * 100.0


def trend_structure(rets: Dict[str, float]) -> str:
    r10, r30, r90 = rets["R10"], rets["R30"], rets["R90"]
    if r10 > 0 and r30 > 0 and r90 > 0:
        return "UP"
    if r10 < 0 and r30 < 0 and r90 < 0:
        return "DOWN"
    if r90 > 0 and r30 > 0 and r10 < 0:
        return "DIP"
    return "MIXED"


def allocation_adjustment_factor(ticker: str) -> float:
    """
    Pousse lentement vers la cible :
      diff = target - current (en points de %)
      factor ≈ 1 + 0.03 * diff, borné.
    """
    target = PORTFOLIO_TARGETS.get(ticker)
    current = CURRENT_WEIGHTS.get(ticker)
    if target is None or current is None:
        return 1.0

    diff = target - current
    raw = 1.0 + 0.03 * diff
    return max(0.85, min(1.25, raw))


def dca_decision(
    ticker: str,
    rets: Dict[str, float],
    mom: float,
    is_near_ath: Optional[bool],
    price_target_delta: Optional[float],
) -> (str, float):
    """
    Action DCA finale + facteur.
    Combine :
      - structure 10d/30d/90d
      - intensité du momentum
      - prix vs target analystes
      - écart à la cible de portefeuille
      - proximité ATH
    """
    tr = trend_structure(rets)
    action = "MAINTAIN"
    factor = 1.0

    # Base sur la tendance
    if tr in ("UP", "DIP"):
        action, factor = "ACCELERATE", 1.15
    elif tr == "DOWN":
        action, factor = "SLOW DOWN", 0.90

    # Amplification momentum
    if mom >= 20 and tr in ("UP", "DIP"):
        action, factor = "ACCELERATE STRONGLY", 1.30
    elif mom <= -15 and tr == "DOWN":
        action, factor = "SLOW DOWN", 0.85

    # ATH : jamais "strongly" sur ATH
    if is_near_ath and action == "ACCELERATE STRONGLY":
        action, factor = "ACCELERATE", 1.15

    # Prix vs target
    if price_target_delta is not None:
        if price_target_delta > 15.0 and "ACCELERATE" in action:
            # Trop cher → prudence
            action, factor = "MAINTAIN", 1.0
        elif price_target_delta < -15.0 and tr != "DOWN":
            # Pas cher et pas en vraie tendance baissière → plus agressif
            if action == "MAINTAIN":
                action, factor = "ACCELERATE", 1.15
            elif action == "ACCELERATE":
                action, factor = "ACCELERATE STRONGLY", 1.30

    # Ajustement allocation long terme
    factor *= allocation_adjustment_factor(ticker)

    # Bornes
    factor = max(0.75, min(1.40, factor))
    factor = round(factor, 2)
    return action, factor


def distribute_budget(budget: float, factors: Dict[str, float]) -> Dict[str, int]:
    """
    Convertit les facteurs DCA en montants en dollars.
    Montants arrondis au 10 $ le plus proche.
    """
    if not factors:
        print("⚠️ No valid DCA factors, no allocation computed.")
        return {}

    total = sum(factors.values())
    if total <= 0:
        print("⚠️ Sum of DCA factors <= 0, no allocation.")
        return {}

    raw = {t: budget * f / total for t, f in factors.items()}
    alloc = {t: int(round(v / 10.0) * 10) for t, v in raw.items()}

    diff = int(round(budget - sum(alloc.values())))
    if diff and alloc:
        first = next(iter(alloc))
        alloc[first] += diff

    return alloc

# ==========================
# HTML RENDERING
# ==========================

STYLE = """
<style>
body{font-family:Arial,Helvetica,sans-serif;background:#f3f4f6;margin:0;padding:0;}
h2,h3{color:#2c3e50;}
.container{padding:16px;}
.box{background:#ffffff;padding:12px 16px;margin-bottom:16px;border-radius:8px;
     box-shadow:0 2px 6px rgba(0,0,0,0.06);}
table{border-collapse:collapse;font-size:13px;width:100%;}
th,td{border:1px solid #dde2eb;padding:6px 8px;}
th{background:#2c3e50;color:#ffffff;text-align:left;}
tr.green{background:#d4efdf;}
tr.red{background:#fdecea;}
tr.neutral{background:#f8f9fa;}
.grid-2{display:grid;grid-template-columns:1.1fr 0.9fr;gap:16px;}
.grid-3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:16px;}
.bar{height:16px;border-radius:6px;background:#e5e8ef;overflow:hidden;}
.fill{height:100%;background:linear-gradient(90deg,#2ecc71,#1abc9c);}
.tip-title{font-weight:bold;margin-bottom:4px;}
</style>
"""

def row_color(mom: float) -> str:
    if mom >= 15.0:
        return "green"
    if mom <= -10.0:
        return "red"
    return "neutral"


def build_html_report(
    budget: float,
    snapshots: Dict[str, Dict],
    dca: Dict[str, Dict],
    amounts: Dict[str, int],
    monthly_tip: str,
    per_ticker_advice: Dict[str, str],
) -> str:
    # ---- Multi-horizon + momentum ----
    perf_rows = []
    for t, info in snapshots.items():
        mom = info["momentum"]
        cls = row_color(mom)
        perf_rows.append(
            f"<tr class='{cls}'>"
            f"<td>{t}</td>"
            f"<td>{info['R10']:.2f}%</td>"
            f"<td>{info['R30']:.2f}%</td>"
            f"<td>{info['R90']:.2f}%</td>"
            f"<td>{mom:.1f}%</td>"
            "</tr>"
        )
    multi_table = (
        "<table>"
        "<tr><th>Ticker</th><th>10d</th><th>30d</th><th>90d</th><th>Momentum</th></tr>"
        + "".join(perf_rows)
        + "</table>"
    )

    # ---- DCA table ----
    dca_rows = []
    for t, info in dca.items():
        act = info["action"]
        factor = info["factor"]
        amt = amounts.get(t, 0)
        dca_rows.append(
            "<tr>"
            f"<td>{t}</td>"
            f"<td>{act}</td>"
            f"<td>×{factor:.2f}</td>"
            f"<td>${amt}</td>"
            "</tr>"
        )
    dca_table = (
        "<table>"
        "<tr><th>Ticker</th><th>Action</th><th>Factor</th><th>Amount</th></tr>"
        + "".join(dca_rows)
        + "</table>"
    )

    # ---- Market price vs analyst target ----
    price_rows = []
    for t, info in snapshots.items():
        price = info["price"]
        target = info["target"]
        delta = info["delta"]
        if price is None and target is None:
            continue
        price_str = "n/a" if price is None else f"${price:,.2f}"
        target_str = "n/a" if target is None else f"${target:,.2f}"
        delta_str = "n/a" if delta is None else f"{delta:+.1f}%"
        price_rows.append(
            "<tr>"
            f"<td>{t}</td>"
            f"<td>{price_str}</td>"
            f"<td>{target_str}</td>"
            f"<td>{delta_str}</td>"
            "</tr>"
        )
    price_table = (
        "<table>"
        "<tr><th>Ticker</th><th>Price</th><th>Target</th><th>Delta</th></tr>"
        + "".join(price_rows)
        + "</table>"
    )

    # ---- Module X : progression allocation ----
    alloc_rows = []
    for t, target_w in PORTFOLIO_TARGETS.items():
        cur = CURRENT_WEIGHTS.get(t, 0.0)
        tgt = float(target_w)
        progress = int(min(100.0, (cur / tgt * 100.0) if tgt > 0 else 0.0))
        alloc_rows.append(
            "<tr>"
            f"<td>{t}</td>"
            f"<td>{cur:.1f}%</td>"
            f"<td>{tgt:.1f}%</td>"
            f"<td><div class='bar'><div class='fill' "
            f"style='width:{progress}%;'></div></div> {progress}%</td>"
            "</tr>"
        )
    module_x_table = (
        "<table>"
        "<tr><th>Ticker</th><th>Current</th><th>Target</th><th>Progress</th></tr>"
        + "".join(alloc_rows)
        + "</table>"
    )

    # ---- How to read ----
    how_to_read = """
<ul>
  <li><strong>Momentum</strong> = 50% × 10d + 30% × 30d + 20% × 90d.</li>
  <li><strong>ACCELERATE</strong> = tendance saine ou dip contrôlé.</li>
  <li><strong>ACCELERATE STRONGLY</strong> = surperformance claire (mais rarement si proche de l'ATH).</li>
  <li><strong>SLOW DOWN</strong> = vraie faiblesse structurelle, surtout hors valeurs cœur.</li>
  <li><strong>Price vs Target</strong> décale le curseur (trop cher → prudence, très en dessous → un peu plus agressif si la structure tient).</li>
  <li><strong>Module X</strong> force la convergence lente vers ta cible long terme sans couper le DCA.</li>
</ul>
"""

    # ---- Suggestions par action ----
    advice_items = [
        f"<li><strong>{t}</strong> — {txt}</li>"
        for t, txt in per_ticker_advice.items()
    ]
    advice_block = "<ul>" + "".join(advice_items) + "</ul>"

    now_str = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>AI DCA Report</title>
{STYLE}
</head>
<body>
<div class="container">

  <h2>AI DCA Report — Budget ${budget:.0f}</h2>
  <p style="font-size:12px;color:#7f8c8d;">Generated at {now_str}</p>

  <div class="box">
    <div class="grid-3">
      <div>
        <h3>Multi-horizon performance</h3>
        {multi_table}
      </div>
      <div>
        <h3>Market price vs analyst target</h3>
        {price_table}
      </div>
      <div>
        <h3>Module X — Progression vers allocation cible</h3>
        {module_x_table}
      </div>
    </div>
  </div>

  <div class="box">
    <div class="grid-2">
      <div>
        <h3>DCA recommendation</h3>
        {dca_table}
      </div>
      <div>
        <h3>How to read</h3>
        {how_to_read}
      </div>
    </div>
  </div>

  <div class="box">
    <div class="grid-2">
      <div>
        <h3>Conseil du mois</h3>
        <p class="tip-title">{monthly_tip}</p>
      </div>
      <div>
        <h3>Suggestions par action</h3>
        {advice_block}
      </div>
    </div>
  </div>

</div>
</body>
</html>
"""
    return html

# ==========================
# STRATEGIC ADVICE
# ==========================

def build_per_ticker_advice(
    snapshots: Dict[str, Dict],
    dca: Dict[str, Dict],
) -> Dict[str, str]:
    advice: Dict[str, str] = {}
    for t, snap in snapshots.items():
        mom = snap["momentum"]
        delta = snap["delta"]
        ath = snap["near_ath"]
        act = dca[t]["action"]
        cur = CURRENT_WEIGHTS.get(t, 0.0)
        tgt = PORTFOLIO_TARGETS.get(t, 0.0)
        diff = tgt - cur

        parts = []
        if diff > 2:
            parts.append(f"sous-pondéré (~{diff:.1f} pts sous la cible)")
        elif diff < -2:
            parts.append(f"sur-pondéré (~{abs(diff):.1f} pts au-dessus de la cible)")
        else:
            parts.append("proche de la cible")

        if delta is not None:
            if delta < -15:
                parts.append("prix nettement sous le target des analystes")
            elif delta > 15:
                parts.append("prix nettement au-dessus du target des analystes")
            else:
                parts.append("prix proche des estimations des analystes")

        if ath:
            parts.append("proche de ses plus hauts 5 ans")

        if mom > 15:
            parts.append("momentum fort")
        elif mom < -10:
            parts.append("momentum fragile à court terme")
        else:
            parts.append("momentum modéré")

        parts.append(f"action DCA: {act.lower()}")
        advice[t] = "; ".join(parts) + "."

    return advice


def build_monthly_tip(
    snapshots: Dict[str, Dict],
    dca: Dict[str, Dict],
) -> str:
    """
    Choisit une valeur sous-pondérée avec momentum positif
    comme "meilleur candidat" pour surpondérer légèrement.
    """
    candidates = []
    for t, snap in snapshots.items():
        cur = CURRENT_WEIGHTS.get(t, 0.0)
        tgt = PORTFOLIO_TARGETS.get(t, 0.0)
        diff = tgt - cur
        mom = snap["momentum"]
        delta = snap["delta"]
        act = dca[t]["action"]

        if diff > 1.5 and mom > 0:
            score = diff * 2.0 + mom / 5.0
            candidates.append((score, t, diff, mom, delta, act))

    if not candidates:
        return (
            "Aucune opportunité évidente ne se détache ce mois-ci. "
            "Garde un DCA régulier sur toutes les valeurs cœur et "
            "laisse la convergence vers la cible se faire progressivement."
        )

    candidates.sort(reverse=True)
    _, t, diff, mom, delta, act = candidates[0]

    if delta is None:
        delta_str = "sans données de target"
    elif delta < -10:
        delta_str = "actuellement en dessous des targets des analystes"
    elif delta > 10:
        delta_str = "déjà au-dessus des targets des analystes"
    else:
        delta_str = "proche des targets des analystes"

    return (
        f"Si tu dois légèrement surpondérer une ligne ce mois-ci, {t} est un bon candidat : "
        f"sous-pondéré d’environ {diff:.1f} points vs ta cible, momentum d’environ {mom:.1f}%, "
        f"et {delta_str}. Action DCA proposée : « {act} »."
    )

# ==========================
# MAIN
# ==========================

def run(budget: float) -> None:
    snapshots: Dict[str, Dict] = {}
    dca: Dict[str, Dict] = {}

    for t in TICKERS:
        close = yahoo_close(t)
        if close is None:
            continue

        rets = multi_returns(close)
        mom = momentum_score(rets)
        price, target, delta = price_and_target(t)
        ath = near_ath(t)
        action, factor = dca_decision(t, rets, mom, ath, delta)

        snapshots[t] = {
            "R10": rets["R10"] * 100.0,
            "R30": rets["R30"] * 100.0,
            "R90": rets["R90"] * 100.0,
            "momentum": mom,
            "price": price,
            "target": target,
            "delta": delta,
            "near_ath": ath,
        }
        dca[t] = {"action": action, "factor": factor}

        print(
            f"{t:5s} | 10d {rets['R10']*100:6.2f}%  "
            f"30d {rets['R30']*100:6.2f}%  "
            f"90d {rets['R90']*100:6.2f}%  "
            f"| Momentum {mom:6.2f}% | DCA {action} x{factor:.2f}"
        )

    factors = {t: info["factor"] for t, info in dca.items()}
    amounts = distribute_budget(budget, factors)

    print("\n--- DCA amounts ---")
    for t, amt in amounts.items():
        print(f"{t:5s} -> ${amt}")

    per_ticker_advice = build_per_ticker_advice(snapshots, dca)
    monthly_tip = build_monthly_tip(snapshots, dca)

    html = build_html_report(budget, snapshots, dca, amounts, monthly_tip, per_ticker_advice)
    out_path = "ai_signals_report.html"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\n✅ Report generated: {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AI DCA helper: momentum, price vs target, allocation and personalised suggestions."
    )
    parser.add_argument(
        "--budget",
        type=float,
        default=DEFAULT_BUDGET,
        help=f"Monthly budget in USD (default: {DEFAULT_BUDGET})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(args.budget)


if __name__ == "__main__":
    main()
