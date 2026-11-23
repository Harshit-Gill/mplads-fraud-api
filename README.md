# MPLADS Fraud & Anomaly Detection (SIH26102)

Flags suspicious works entries in the **Member of Parliament Local Area Development Scheme (MPLADS)** so auditors know where to look first. Built for Smart India Hackathon problem statement SIH26102 (MoSPI).

- **Dashboard:** https://harshit-gill.github.io/mplads-fraud-api/ — a live risk scorer for new entries, plus an auditor view of all 8,658 scored works
- **API:** FastAPI on Render, at `POST /predict` (docs at `/docs`)

## How it works

Real audit data has no fraud labels, so the detector is built **without using labels**. Labels in the synthetic dataset are used only once, at the end, to measure performance on a held-out test set.

1. **Five audit rules**, each with a threshold from domain facts or label-free statistics, never tuned on the labels:

| Rule | Fires when | Role |
|---|---|---|
| Cost overrun | expenditure > sanctioned amount | flags on its own |
| Fiscal year-end dumping | sanctioned in Feb–Mar with < 5% spent | flags on its own |
| Fund splitting | ≥ 3 works for the same vendor–MP pair, each just under the ₹2.5 crore installment | flags on its own |
| Implausibly fast completion | completed faster than the bottom 1% of all completion times | flags on its own |
| Round-number invoice | expenditure is an exact multiple of ₹10,000 | weak signal: adds to the risk score only |

2. **Isolation Forest** finds multivariate outliers (vendor collusion, price inflation) from 11 engineered features, such as a vendor's spending deviation within a category, how many MPs a vendor works with, and utilization ratio. Contamination is set to 0.07 as a stated prior (single-digit irregularity rates in audit literature), not the true anomaly rate.

3. **Final flag** = any strong rule OR Isolation Forest. **Risk score (0–100)** = the highest rule severity or the scaled Isolation Forest score.

## Results (held-out test set: 2,165 works, 252 anomalies, 11.6% base rate)

| Version | Works flagged | Precision | Recall | F1 |
|---|---|---|---|---|
| v2.0 | 46% | 20% | 78% | 0.31 |
| **v2.1 (current)** | **13%** | **47%** | **54%** | **0.50** |

**What changed in v2.1:**
- **The round-number rule caused most false alarms.** It fired on 447 test works, and only 7% of them were anomalous, which is below the base rate. It now only raises the risk score and can no longer flag a work on its own.
- **Evaluation and deployment now use the same Isolation Forest setting.** v2.0 evaluated with `contamination="auto"` but deployed with 0.07.

The result: an auditor reviewing flagged works now finds a real anomaly about every second case, instead of every fifth. The cost is lower recall on the subtler anomaly types.

Recall by anomaly type (v2.1):

| Anomaly type | Caught |
|---|---|
| Cost overrun | 45 / 45 (100%) |
| Fiscal year-end dumping | 42 / 42 (100%) |
| Vendor collusion | 19 / 48 (40%) |
| Price inflation | 10 / 31 (32%) |
| Implausibly fast completion | 11 / 34 (32%) |
| Round-number invoice | 3 / 15 (20%) |
| Fund splitting | 6 / 37 (16%) |

## Limitations

- **Synthetic data.** 8,658 works with 7 planted anomaly types. Real MPLADS records will be messier.
- **Cold start.** A vendor or MP never seen in training falls back to category or global averages.
- **Fund splitting has low recall** because the rule only looks at the ₹2.5 crore installment band. Lowering the band to catch more would mean tuning on labels.
- **Screening aid, not proof of fraud.** A flag means "review this", not "this is fraud".

## Files

| File | Purpose |
|---|---|
| `mplads_fraud_detection_v2.py` | Features, rules, settings and hold-out evaluation; writes `eval_metrics.json` |
| `train_and_export.py` | Trains the production model on all data; writes `iso_model.joblib`, `artifacts.json`, `scored_works.json` |
| `main.py` | FastAPI app: `POST /predict`, `GET /known-values`, `GET /` |
| `index.html` | Dashboard (GitHub Pages); loads `scored_works.json` and `eval_metrics.json` |
| `mplads_dataset_v2.csv` | Synthetic dataset |

## Run locally

```bash
pip install -r requirements.txt
python mplads_fraud_detection_v2.py   # evaluate on hold-out, write eval_metrics.json
python train_and_export.py            # train production model + dashboard data
uvicorn main:app --reload             # API at http://127.0.0.1:8000/docs
```

## Deploy (Render)

- **Build command:** `pip install -r requirements.txt`
- **Start command:** `uvicorn main:app --host 0.0.0.0 --port $PORT`

`scikit-learn` is pinned because `iso_model.joblib` must be loaded with the version that saved it.
