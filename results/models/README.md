# results/models/ — Saved Model Files

This directory is reserved for serialized trained model objects.

Currently empty — models are retrained on each pipeline run.
To save models, add joblib.dump() calls to the relevant phase scripts.

Example (add to phase11_classical_models.py or phase15_vqc.py):
`python
import joblib
joblib.dump(rf_model, 'results/models/random_forest_best.pkl')
joblib.dump(lr_model, 'results/models/logistic_regression_best.pkl')
# VQC model can be saved via:
optimal_vqc.vqc.save('results/models/vqc_optimal.model')
`

---

*NIFTY50-VQC Project | results/models/ | Last updated: 2026-10-03*
