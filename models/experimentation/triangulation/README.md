# EC3D triangulation learning playground

This directory is a guided, deliberately incomplete implementation exercise.
The EC3D loader and synthetic camera scene are provided. You implement the
geometry yourself, one small function at a time.

## Safety and environment

`data.pickle` is about 917 MiB and takes roughly a minute to load. Python pickle
files can execute code while loading, so use only the trusted local EC3D file.
The synthetic lessons do not load it.

Run commands from the SquatSpot project root with the existing environment:

```powershell
.\venv\Scripts\python.exe models\experimentation\triangulation\learning_checks.py
```

## Mental model

A calibrated camera maps a homogeneous world point into an image:

```text
X_world (4-vector) ── P = K [R | t] ──> x_image (3-vector) ── divide by z ──> (u, v)
```

Triangulation reverses that mapping using observations of the same joint from
two or more cameras. Each observation supplies two linear constraints. DLT
stacks those constraints and solves for the homogeneous 3D point using SVD.

## Lessons

### 1. Inspect the dataset

```powershell
.\venv\Scripts\python.exe models\experimentation\triangulation\inspect_ec3d.py
```

Study:

- the four camera IDs;
- `K`, distortion coefficients, `R`, and `t`;
- one synchronized frame's `2D_op` and `3D_gt` dictionaries;
- joint-name overlap across views.

This loads the full pickle. Do it once per learning session.

### 2. Build a projection matrix

Open `geometry.py` and implement `projection_matrix()`.

Given `K` shaped `(3, 3)`, `R` shaped `(3, 3)`, and `t` shaped `(3,)`, produce:

```text
P = K [R | t]
```

Then implement `project_point()` and run the synthetic checks.

### 3. Triangulate one point with DLT

Implement `triangulate_point_dlt()` without calling OpenCV's triangulation
function. For each camera observation `(u, v)`, append:

```text
u * P[2] - P[0]
v * P[2] - P[1]
```

Solve `A X = 0` with `numpy.linalg.svd`. The last row of `Vᵀ` is the homogeneous
solution. Divide by its fourth coordinate.

### 4. Measure reprojection error

Implement `reprojection_errors()`. A correct synthetic reconstruction should
have nearly zero error when observations are noise-free. Add pixel noise and
observe how baseline, camera count, and geometry affect the result.

### 5. Move to EC3D

Implement the marked functions in `triangulate_ec3d.py`:

1. Convert the stored camera parameters to projection matrices.
2. Establish common joint names across at least two `2D_op` views.
3. Confirm whether stored 2D points need lens undistortion.
4. Triangulate each common joint.
5. Compare it with `3D_gt`.

Start with one frame and cameras `6_1` and `6_2`. Only then extend to all four
cameras and a sequence.

## Evaluation checklist

For each reconstructed joint report:

- number of contributing views;
- reprojection error per camera in pixels;
- Euclidean 3D error against `3D_gt`;
- whether the point lies in front of every contributing camera.

For a sequence report:

- triangulation coverage;
- MPJPE;
- median and P95 3D error;
- mean reprojection error;
- per-joint error;
- failure counts by number of available views.

Do not judge correctness from a visually plausible skeleton alone. A sign,
axis, unit, or extrinsic-convention error can still produce a recognizable
shape.

## Suggested supervision loop

After each lesson, share your implementation or check output. We can review the
math and diagnostics before you continue. Avoid implementing the entire
pipeline at once; projection and reprojection should pass before DLT is trusted.

