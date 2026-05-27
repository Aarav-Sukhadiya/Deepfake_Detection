# dfdetect samples

Two **synthetic** placeholder PNGs for end-to-end smoke testing of the
pipeline. They are not real faces and not real deepfakes — they only exist so
`dfdetect analyze samples/*.png` exercises the full input → preprocess →
infer → report path on a fresh checkout.

- `real_face.png` — solid ellipse "face" on a smooth background.
- `fake_face.png` — the same layout with high-frequency noise and a slightly
  misaligned mouth.

Replace these with your own labelled fixtures once you wire in a real
deepfake-trained checkpoint; predictions on the synthetic pair are not
meaningful for verifying detection quality.
