# Tests and evidence

## Automated Phase 1 checks

From the project root:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m scripts.check_environment
.\.venv\Scripts\python.exe -m ml.audit_dataset
```

Unit tests use synthetic geometric fixtures to check normalization, shape, masks, identifier validation, quality rejection, safe clip round-trip, duplicate detection, consent requirements and timestamps. They are not sign-language training data and must not be reported as recognition accuracy. The environment script also executes Google's actual detector on a blank frame and checks that it returns no hands.

## Local camera acceptance checklist

Record the actual date, camera, operator pseudonym and result in `docs/LOCAL_RESULTS.csv`. Do not prefill PASS before testing.

1. Preview opens and closes with Q and the window close button.
2. One hand and two hands display appropriate landmark overlays.
3. No hands shows zero; partial/crossed hands do not crash the app.
4. A human-reviewed label can record; an unreviewed label is blocked.
5. SPACE counts down and produces one complete numeric clip.
6. Short/incomplete or poorly detected clips are rejected.
7. Ending during a clip discards the partial recording.
8. Audit counts real clips and detects a deliberately copied duplicate.
9. Multiple pseudonymous signers and genuine separate sessions are recorded.
10. No JPG, video or audio appears in the raw data folder.

## Later recognition and interaction checklist

Only after a model exists, measure correct/incorrect alphabets, low-confidence cases, unknown gestures, long holds, repeated letters, transition frames, lighting, angles, position and camera distance. Hold out participants and sessions. Report support counts, failures and non-results; never insert invented accuracy figures. Also test edits, undo, prefix suggestions, personalized vocabulary, cross-user history isolation, sentence editing, speech playback and practice records against real services.

An automated speech button test cannot prove that sound was audible; record a local listening test. Browser camera permissions and MongoDB persistence require actual integration tests. A phase is complete only when its evidence exists.
