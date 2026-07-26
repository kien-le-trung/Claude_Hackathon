# MM-Fit / MediaPipe 3D reconstruction playground

This directory is an intentionally small playground for measuring MediaPipe
Pose depth accuracy against MM-Fit session `w00`. It is separate from runtime
application code so experiments cannot affect the API.

## What MediaPipe's `z` means

MediaPipe returns two different 3D representations:

- `pose_landmarks`: `x` and `y` are image-normalized. `z` is relative depth
  with the midpoint of the hips near the origin; its magnitude uses roughly
  the same scale as normalized `x`. Smaller values are closer to the camera.
  It is not an absolute metric camera-space depth.
- `pose_world_landmarks`: hip-centered coordinates in meters. These are the
  appropriate starting point for metric 3D skeleton comparison, but they are
  still monocular estimates and do not contain the subject's absolute
  translation from the camera.

Therefore, do not compare either MediaPipe `z` directly with the raw MM-Fit
coordinate column. First establish axes, units, joint correspondence, and a
per-frame or sequence-level alignment. Report normalized-landmark depth and
world-landmark accuracy as separate experiments.

## Confirmed `w00` alignment facts

Run:

```powershell
.\venv\Scripts\python.exe models\experimentation\3d_reconstruction\alignment.py
```

Current files show:

- RGB: 69,796 frames at 30 FPS (about 2,326.53 seconds).
- Pose 3D: shape `(3, 63918, 18)`.
- Pose row zero is a repeated frame ID, not a joint.
- Embedded pose frame IDs are contiguous from 4050 through 67967.
- NumPy pose row `i` maps to RGB frame `4050 + i`.
- The first squat label is frames 4040–4500, so comparable ground truth begins
  at frame 4050.

Always join by the embedded frame ID. Never assume pose row index equals video
frame index.

## Playground layout

- `config.py`: paths, MM-Fit anatomy, MediaPipe comparison joints, and output
  schema constants.
- `alignment.py`: working loaders and alignment diagnostics.
- `extract_mediapipe.py`: manual implementation target for extracting both
  MediaPipe representations from aligned `w00` squat frames.
- `evaluate.py`: manual implementation target for coordinate alignment and
  accuracy metrics.
- `tests/test_alignment.py`: executable checks protecting the frame-ID rule.
- `outputs/`: ignored generated artifacts.

## Suggested implementation order

1. Run `alignment.py` and the alignment tests.
2. Implement `extract_mediapipe.py`. Initially process only the intersection of
   squat label ranges and available pose frame IDs. Preserve the original RGB
   frame ID in every output record.
3. Visually overlay MediaPipe `x/y` landmarks on a few RGB frames and compare
   squat phase against MM-Fit before measuring 3D error.
4. Implement the common-joint mapping in `evaluate.py`.
5. Infer MM-Fit axes and units from evidence. Center both skeletons at the
   pelvis, then use a similarity/Procrustes alignment to handle rotation,
   reflection checks, and scale.
6. Report MPJPE, PA-MPJPE, per-joint error, and depth-axis error. Include
   detection coverage so missing poses cannot silently improve the score.
7. Compare sequence-level alignment with per-frame alignment. Per-frame
   Procrustes is useful diagnostically but can hide temporal depth instability.

## Proposed extraction artifact

Write `outputs/w00_mediapipe.npz` with:

- `frame_ids`: `(N,)` original RGB frame IDs.
- `normalized_landmarks`: `(N, 33, 5)` storing `x, y, z, visibility, presence`.
- `world_landmarks`: `(N, 33, 5)` with the same field order.
- `detected`: `(N,)` boolean detection mask.

Keep missing detections as `NaN` rows rather than dropping them. This preserves
alignment and makes coverage measurable.

## Evaluation cautions

- MM-Fit has 17 anatomical joints; MediaPipe has 33 landmarks. Only compare
  defined anatomical correspondences.
- Pelvis can be constructed as the midpoint of MediaPipe hips. Neck and head
  require documented derived definitions if included.
- Determine whether MM-Fit coordinates are millimeters before converting to
  meters; do not encode that assumption without checking.
- Check left/right orientation and possible reflection explicitly.
- Split results by squat range and movement phase; a single aggregate error can
  conceal systematic depth failures.

