# AI Signals Helper (Yahoo + Stooq, GitHub Actions Ready)

This repository contains a small Python tool that:
- Fetches recent daily prices for a set of AI-related tickers
  using **Yahoo Finance** as primary source and **Stooq** as backup.
- Computes **5-trading-day performance** for each ticker.
- Classifies each ticker into one of three signals:
  - `PANIC_BUY`  → strong recent drop (potential buying opportunity)
  - `EUPHORIA`   → strong recent rally (be careful / consider delaying)
  - `NORMAL`     → nothing special
- Proposes a **simple allocation suggestion** for a given monthly budget.
- Can optionally **send the full report by email**, with flexible rules on
  when emails should be sent.
- Is ready to be run as a **GitHub Action** on a schedule or on demand.

> ⚠️ This is **not** financial advice. It is a helper / monitoring tool.  
> You are always responsible for your own investment decisions.

---

## 1. Project structure

```text
.
├── ia_signals.py              # Main Python script
├── requirements.txt           # Python dependencies
└── .github/
    └── workflows/
        └── ai-signals.yml     # GitHub Actions workflow
```

---

## 2. Python script overview

The core logic lives in `ia_signals.py`. Main features:

- **Tickers configuration**: in the `TICKERS` list at the top of the file.
- **5-day performance**: using adjusted daily closes from Yahoo / Stooq.
- **Robustness**:
  - Yahoo is queried first.
  - Stooq is used as backup and cross-check.
  - If both are available and differ by more than `MISMATCH_TOLERANCE`,
    the ticker is marked as `MISMATCH`.
- **Signals**:
  - `PANIC_BUY`  if 5d change ≤ `PANIC_THRESHOLD` (e.g. -8%).
  - `EUPHORIA`   if 5d change ≥ `EUPHORIA_THRESHOLD` (e.g. +10%).
  - `NORMAL`     otherwise.
- **Allocation suggestion** given a budget:
  - If there are `PANIC_BUY` tickers:
    - 70% of the budget split equally across `PANIC_BUY`
    - 30% of the budget split equally across `NORMAL`
    - 0 allocated to `EUPHORIA`
  - Else if only `NORMAL`:
    - 100% of the budget split equally across `NORMAL`
  - Else:
    - 0 allocated, everything kept as cash.

At the end, the script prints:
- per-ticker 5d performance & signal,
- the suggested allocation split,
- any cash remainder.

---

## 3. Email notifications

The script can send an email report using **SMTP**, configured via
environment variables.

### Required environment variables

- `SMTP_HOST` – SMTP server hostname (e.g. `smtp.gmail.com`)
- `SMTP_PORT` – SMTP port (e.g. `587`)
- `SMTP_USER` – SMTP username / login
- `SMTP_PASSWORD` – SMTP password or app-specific password
- `EMAIL_FROM` – sender email address
- `EMAIL_TO` – recipient email address

If any of these are missing, the script will **skip** sending email
and print a message.

### Email mode

The logic to decide *when* to send an email is controlled by:

- the command-line argument `--email-mode`, and/or
- the environment variable `EMAIL_MODE`.

Accepted values:

- `only_on_action` (default)  
  Send email **only if**:
  - at least one ticker is `PANIC_BUY` OR
  - at least one ticker is `EUPHORIA` OR
  - there is a data `MISMATCH` / `NO_DATA` OR
  - an internal error occurred.
- `always`  
  Always send an email, even if all signals are `NORMAL`.
- `never`  
  Never send an email.

If both are set, the environment variable `EMAIL_MODE` overrides the
default, but you can still override via `--email-mode` when running
manually.

Example manual run:

```bash
export SMTP_HOST="smtp.example.com"
export SMTP_PORT="587"
export SMTP_USER="user@example.com"
export SMTP_PASSWORD="secret"
export EMAIL_FROM="bot@example.com"
export EMAIL_TO="you@example.com"
export EMAIL_MODE="only_on_action"

py ia_signals.py --budget 4000
```

Or force an email every time:

```bash
EMAIL_MODE="always" python ia_signals.py --budget 4000
```

Or disable emails completely for a test run:

```bash
EMAIL_MODE="never" python ia_signals.py --budget 4000 --dry-run
```

Note: `--dry-run` allows you to see what would happen, without actually
sending any email.

---

## 4. Running locally

1. Create a virtual environment (optional but recommended):

   ```bash
   py -m venv .venv
   source .venv/bin/activate  # on Windows: .venv\\Scripts\\activate
   ```

2. Install dependencies:

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. Run the script:

   ```bash
   py ia_signals.py --budget 4000
   ```

   You should see something like:

   ```text
   === AI Robust Signals 2025-11-28 18:00 UTC ===

   NVDA     | 5d:  -9.30% | Source: YAHOO    | Signal: PANIC_BUY
   MSFT     | 5d:   2.10% | Source: YAHOO    | Signal: NORMAL
   QQQ      | 5d:  -5.20% | Source: YAHOO    | Signal: PANIC_BUY

   --- Allocation suggestion ---
     NVDA     -> 1400.00 USD
     QQQ      -> 1400.00 USD
     MSFT     -> 1200.00 USD
   ```

---

## 5. GitHub Actions integration

The workflow file `.github/workflows/ai-signals.yml` runs the script
automatically:

- on a schedule (weekdays at 18:00 UTC),
- and manually via the **"Run workflow"** button in GitHub.

```yaml
name: AI Signals Daily

on:
  workflow_dispatch:
  schedule:
    # Runs at 18:00 UTC Monday to Friday
    - cron: "0 18 * * 1-5"

jobs:
  ai-signals:
    runs-on: ubuntu-latest

    steps:
      - name: Checkout repository
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt

      - name: Run AI signals script
        env:
          SMTP_HOST: ${{ secrets.SMTP_HOST }}
          SMTP_PORT: ${{ secrets.SMTP_PORT }}
          SMTP_USER: ${{ secrets.SMTP_USER }}
          SMTP_PASSWORD: ${{ secrets.SMTP_PASSWORD }}
          EMAIL_FROM: ${{ secrets.EMAIL_FROM }}
          EMAIL_TO: ${{ secrets.EMAIL_TO }}
          # Optional: override email mode here (only_on_action, always, never)
          EMAIL_MODE: ${{ secrets.EMAIL_MODE }}
        run: |
          python ia_signals.py --budget 4000
```

### 5.1. Setting GitHub Secrets

In your repository on GitHub:

1. Go to **Settings → Secrets and variables → Actions**.
2. Add the following **Repository secrets**:
   - `SMTP_HOST`
   - `SMTP_PORT`
   - `SMTP_USER`
   - `SMTP_PASSWORD`
   - `EMAIL_FROM`
   - `EMAIL_TO`
   - (optional) `EMAIL_MODE` → e.g. `only_on_action`, `always`, or `never`.

Once secrets are configured, the workflow can send email through your SMTP
provider whenever the script decides it should.

---

## 6. Customizing tickers and thresholds

In `ia_signals.py` near the top, you will find:

```python
TICKERS: List[str] = [
    "NVDA",
    "MSFT",
    "GOOGL",
    "META",
    "AMZN",
    "ASML",
    "QQQ",
]

PANIC_THRESHOLD = -0.08   # -8% over last 5 sessions
EUPHORIA_THRESHOLD = 0.10 # +10% over last 5 sessions
```

You can:
- add or remove tickers as you wish,
- make thresholds more or less aggressive (e.g. -0.05 / 0.08).

If you use non-US tickers, you might also want to adjust `STOOQ_OVERRIDE`
so that Stooq symbols match correctly (for example `"TSLA.US"`, `"ASML.NL"`,
etc.).

---

## 7. Safety and limitations

- Data comes from **free public sources** (Yahoo Finance and Stooq).  
  They can experience occasional delays, changes, or outages.
- Signals are based on **past 5 days performance only**; this is *not*
  a full trading system.
- Network issues or API changes may cause temporary errors; the script will
  try to report them clearly and can still send an email warning if
  `only_on_action` mode is enabled.
- The allocation suggestion is **purely mechanical** and does not take into
  account:
  - your current holdings,
  - taxes,
  - transaction fees,
  - your risk profile.

Use this tool as a **helper** to support your own judgment, not as an
automated trading system.

---

## 8. License

You can treat this as boilerplate / template code for your personal use.
No specific license is enforced here; feel free to adapt it for your
private projects.
