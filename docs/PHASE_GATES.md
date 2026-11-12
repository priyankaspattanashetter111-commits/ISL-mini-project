# Twelve phase gates

1. Environment and collection: import/inference checks pass; camera preview is checked locally; real consented clips, reviewed mappings and an audit exist. Current code implements this phase; the local human checks remain pending.
2. Preprocessing: inspect actual inventory; normalize consistently; split signers/sessions without overlap; save NPZ plus manifests; reject leakage and invalid samples.
3. Training: train the agreed static baseline on real features; save model, label mapping, configuration and run metadata. Record actual loss/accuracy curves. Optional LSTM uses genuine temporal data.
4. Evaluation: report actual held-out accuracy, per-class precision/recall/F1, confusion matrix, support and conditions. Fix data or definitions before declaring model readiness.
5. Webcam inference: run the saved model locally; log actual predictions, confidence and latency. Check unknown gestures, missing hands and low confidence.
6. Letter segmentation: test hold/release, long holds, transition noise, repeated letters, space/backspace/replace/undo and low-confidence confirmation. Report assisted boundaries honestly.
7. Backend: build real FastAPI inference and Node application endpoints; test request IDs, validation, errors, rate/size limits, stream cleanup and model-unavailable behavior.
8. Database: connect to actual MongoDB; test collection validators, indexes, ownership and persistence after restart.
9. React: integrate the real APIs and browser camera; verify permissions, responsive layout, keyboard interactions and visible state.
10. Assistant: integrate prefix suggestions, custom vocabulary, history, practice and browser speech with tests. Do not relabel dictionary lookup as an advanced AI model.
11. Integration: run the complete workflow across services and MongoDB; test outage/recovery and concurrent user isolation.
12. Demo: repeat a published checklist on actual cameras and conditions; package source, trained artifacts, results, limitations and commands.

Do not advance a major stage on fabricated successful outputs. A training pipeline that runs on synthetic geometry is a software check only, never an ISL recognition result.
