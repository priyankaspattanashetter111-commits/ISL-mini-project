# Collection schema and normalization

Each compressed NumPy clip has these arrays. Load with `np.load(path, allow_pickle=False)`; never load arbitrary pickle files.

| Array | Shape | Meaning |
|---|---|---|
| points | T x 2 x 21 x 3 | MediaPipe image-space x/y/z for Left and Right slots |
| masks | T x 2 | 1 if that hand was detected; otherwise 0 |
| features | T x 128 | 126 normalized coordinates plus 2 presence flags |
| timestamps_ms | T | Strictly increasing capture timestamps |
| metadata | scalar string | JSON mapping, signer, conditions and quality fields |

T is variable because actual camera and inference speeds differ. A future LSTM input must resample by timestamps to a fixed sequence length, preserving missing-hand masks. Do not silently repeat a still frame and call that temporal training.

For each frame, convert x and z to image-height units with width/height. Compute a common palm scale from wrist-to-middle-knuckle distances of visible hands. Subtract a common x/y wrist origin and divide by that scale. For z, subtract each hand's own wrist z before scaling: image-space z is not a calibrated shared depth across hands. Keep orientation; rotations can change the sign. Relative hand x/y positions remain represented. Handedness slots reduce ordering changes, but tracking errors and hand crossover remain concerns to test.

Both training preprocessing and inference must call exactly this implementation and check `feature_version`. If the mapping or feature definition changes, version it and regenerate or recollect compatible data. A missing hand has zero coordinates plus mask=0; do not pretend an undetected hand is a valid observed neutral gesture.

The collection quality filter counts frames with the required hand count. It cannot verify the right gesture. Human review is still needed. A static sample and a dynamic sample must not be mixed under one variant without an explicit modelling decision.
