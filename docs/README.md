# docs/ — Project Documentation

This directory contains all research documentation for the NIFTY50-VQC project.

---

## Document Index

| File | Purpose |
|------|---------|
| final_report.md | Complete research report: problem, method, results, conclusion |
| methodology.md | Detailed experimental methodology — data, features, models, evaluation |
| experiment_log.md | Phase-by-phase experiment log (EXP-001 to EXP-016) with actual results |
| environment_info.txt | System info, package versions, phase completion status |

---

## Reading Order (recommended)

1. **final_report.md** — Start here for the full story and results
2. **methodology.md** — Deep-dive into the experimental design
3. **experiment_log.md** — Trace exactly what was run and what happened
4. **environment_info.txt** — Reproduce the exact environment

---

## Quick Stats (from docs)

- **Data:** 2,706 NIFTY 50 trading days (2015-2025)
- **Split:** Train 1,963 / Val 495 / Test 248 (chronological)
- **Features:** 8 technical indicators -> PCA to 4 components (94.89% variance)
- **VQC result:** 48.39% Test Accuracy, F1=0.6257 (4-Qubit ZZ+SPSA)
- **Best classical:** Random Forest 52.42% Test Accuracy
- **Statistical test:** No significant difference between any models (McNemar p > 0.50)

---

*NIFTY50-VQC Project | docs/ | Last updated: 2026-10-03*
