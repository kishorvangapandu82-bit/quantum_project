"""
prediction_server.py  —  NIFTY50-VQC Live Prediction API
=========================================================
Start: .venv\Scripts\python.exe prediction_server.py
API:   http://localhost:5050
"""

import os, json, time, datetime, warnings
import numpy as np
import pandas as pd
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS

warnings.filterwarnings("ignore")
app = Flask(__name__)
CORS(app)

MD = "results/models"

# ── Load classical models + scalers at startup ─────────────
print("[Startup] Loading scalers and classical models...")
ss       = joblib.load(f"{MD}/scaler_standard.pkl")
pca_mdl  = joblib.load(f"{MD}/pca_model.pkl")
mms_pca  = joblib.load(f"{MD}/scaler_pca_quantum.pkl")
lr_m     = joblib.load(f"{MD}/model_lr.pkl")
svm_m    = joblib.load(f"{MD}/model_svm.pkl")
rf_m     = joblib.load(f"{MD}/model_rf.pkl")
hist_df  = pd.read_csv(f"{MD}/feature_history.csv", index_col=0, parse_dates=True)

with open(f"{MD}/metadata.json") as f:
    meta = json.load(f)
FEATURE_COLS = [str(c) for c in meta["feature_cols"]]
print(f"[Startup] Classical models ready. Features: {FEATURE_COLS}")

# ── Lazy VQC loader ────────────────────────────────────────
_vqc = None
_vqc_ready = False

def try_load_vqc():
    global _vqc, _vqc_ready
    if _vqc_ready:
        return True
    vqc_path = f"{MD}/model_vqc_params.npy"
    if not os.path.exists(vqc_path):
        return False
    try:
        from qiskit.circuit.library import zz_feature_map, real_amplitudes
        from qiskit.primitives import StatevectorSampler
        from qiskit_machine_learning.algorithms import VQC
        from qiskit_machine_learning.optimizers import SPSA

        weights = np.load(vqc_path)
        vqc = VQC(
            sampler=StatevectorSampler(),
            feature_map=zz_feature_map(feature_dimension=4, reps=1),
            ansatz=real_amplitudes(num_qubits=4, reps=1),
            optimizer=SPSA(maxiter=1),
        )
        # Warm-up to initialise internal circuit
        vqc.fit(np.random.rand(8, 4) * np.pi, np.array([0,1,0,1,0,1,0,1]))
        # Restore trained weights via _fit_result.x (qiskit-ml 0.9.1 API)
        vqc._fit_result.x = weights
        _vqc = vqc
        _vqc_ready = True
        print("[VQC] Loaded successfully!")
        return True
    except Exception as e:
        print(f"[VQC] Load failed: {e}")
        return False

# Try loading VQC right now (will succeed once phase20 finishes)
try_load_vqc()


# ── Feature engineering ────────────────────────────────────
def compute_features(window_df: pd.DataFrame) -> pd.Series:
    df = window_df.copy().sort_index()
    df["daily_return"] = df["close"].pct_change()
    df["sma_5"]        = df["close"].rolling(5).mean()
    df["sma_20"]       = df["close"].rolling(20).mean()
    df["ema_12"]       = df["close"].ewm(span=12, adjust=False).mean()
    df["ema_26"]       = df["close"].ewm(span=26, adjust=False).mean()
    df["volatility_5"] = df["daily_return"].rolling(5).std()
    df["momentum_5"]   = df["close"] - df["close"].shift(5)
    delta  = df["close"].diff()
    gain   = delta.clip(lower=0).rolling(14).mean()
    loss   = (-delta.clip(upper=0)).rolling(14).mean()
    df["rsi_14"] = 100 - 100 / (1 + gain / (loss + 1e-10))
    return df[FEATURE_COLS].iloc[-1]


# ── Run all models ─────────────────────────────────────────
def run_all(X_std, X_q):
    results = {}
    for name, mdl in [("lr", lr_m), ("svm", svm_m), ("rf", rf_m)]:
        p   = int(mdl.predict(X_std)[0])
        prb = mdl.predict_proba(X_std)[0]
        results[name] = {
            "prediction": p,
            "label":      "UP" if p == 1 else "DOWN",
            "confidence": round(float(max(prb)), 4),
            "prob_up":    round(float(prb[1]), 4),
            "prob_down":  round(float(prb[0]), 4),
        }

    # VQC — use if available, else pending
    try_load_vqc()
    if _vqc_ready:
        vp = int(np.array(_vqc.predict(X_q)).ravel()[0])
        try:
            prb = _vqc.predict_proba(X_q)[0]
            conf = round(float(max(prb)), 4)
            p_up = round(float(prb[1]), 4)
            p_dn = round(float(prb[0]), 4)
        except Exception:
            conf, p_up, p_dn = None, None, None

        results["vqc"] = {
            "prediction": vp, "label": "UP" if vp == 1 else "DOWN",
            "confidence": conf, "prob_up": p_up, "prob_down": p_dn,
            "status": "ready",
        }
    else:
        results["vqc"] = {
            "prediction": None, "label": "PENDING",
            "confidence": None, "prob_up": None, "prob_down": None,
            "status": "training — run phase20_save_models.py first",
        }

    # Majority vote (classical only if VQC pending)
    classical_votes = [results["lr"]["prediction"],
                       results["svm"]["prediction"],
                       results["rf"]["prediction"]]
    if results["vqc"]["prediction"] is not None:
        all_votes = classical_votes + [results["vqc"]["prediction"]]
        n = 4
    else:
        all_votes = classical_votes
        n = 3

    maj = 1 if sum(all_votes) > n / 2 else 0
    results["ensemble"] = {
        "prediction": maj,
        "label":      "UP" if maj == 1 else "DOWN",
        "votes_up":   int(sum(all_votes)),
        "votes_down": int(n - sum(all_votes)),
        "total_models": n,
        "vqc_included": _vqc_ready,
    }
    return results


# ── Routes ─────────────────────────────────────────────────
@app.route("/health")
def health():
    return jsonify({
        "status":       "ok",
        "server":       "NIFTY50-VQC Prediction API",
        "time":         datetime.datetime.now().isoformat(),
        "port":         5050,
        "vqc_ready":    _vqc_ready,
        "models_ready": ["LR", "SVM", "RF"] + (["VQC"] if _vqc_ready else []),
    })


@app.route("/predict", methods=["POST", "OPTIONS"])
def predict():
    if request.method == "OPTIONS":
        return "", 200
    try:
        data   = request.get_json(force=True)
        o      = float(data["open"])
        h      = float(data["high"])
        l      = float(data["low"])
        c      = float(data["close"])
        date_s = data.get("date", datetime.date.today().isoformat())
        date   = pd.Timestamp(date_s)

        # Build window: append new OHLCV row to stored history
        new_row   = pd.DataFrame(
            {"open": [o], "high": [h], "low": [l], "close": [c]}, index=[date]
        )
        hist_cols = [col for col in ["open", "high", "low", "close"]
                     if col in hist_df.columns]
        window    = pd.concat([hist_df[hist_cols], new_row]).sort_index().tail(30)
        feats     = compute_features(window)

        if feats.isnull().any():
            missing = feats[feats.isnull()].index.tolist()
            return jsonify({"error": f"Cannot compute features: {missing}. Need more history."}), 400

        X_std = ss.transform(feats.values.reshape(1, -1))
        X_pca = pca_mdl.transform(X_std)
        X_q   = mms_pca.transform(X_pca)
        X_q   = np.clip(X_q, 0, np.pi)  # clip out-of-distribution inputs to valid range

        t0    = time.time()
        preds = run_all(X_std, X_q)
        elapsed = round(time.time() - t0, 3)

        return jsonify({
            "input":       {"date": date_s, "open": o, "high": h, "low": l, "close": c},
            "features":    {k: round(float(v), 4) for k, v in feats.items()},
            "pca_output":  [round(float(v), 4) for v in X_pca[0]],
            "predictions": preds,
            "elapsed_s":   elapsed,
            "vqc_ready":   _vqc_ready,
            "disclaimer":  "Academic research model. NOT financial advice.",
        })

    except KeyError as e:
        return jsonify({"error": f"Missing field: {e}"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/last_known")
def last_known():
    row       = hist_df.iloc[-1]
    close_col = "close" if "close" in hist_df.columns else hist_df.columns[-1]
    return jsonify({
        "date":      str(hist_df.index[-1].date()),
        "close":     round(float(row[close_col]), 2),
        "vqc_ready": _vqc_ready,
    })


@app.route("/model_info")
def model_info():
    return jsonify({
        "feature_cols":    FEATURE_COLS,
        "pca_components":  4,
        "pca_variance":    [50.08, 24.13, 13.17, 7.51],
        "test_accuracies": {
            "VQC": 0.4839 if _vqc_ready else "not ready",
            "LR":  0.5000, "SVM": 0.5000, "RF": 0.5242,
        },
        "training_period": "2015-2022",
        "test_period":     "2025",
        "vqc_ready":       _vqc_ready,
    })


if __name__ == "__main__":
    print("[Server] http://localhost:5050  (Ctrl+C to stop)")
    print(f"[Server] VQC ready: {_vqc_ready}")
    app.run(host="0.0.0.0", port=5050, debug=False)
