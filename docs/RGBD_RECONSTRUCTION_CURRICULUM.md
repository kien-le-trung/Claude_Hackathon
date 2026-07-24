# RGB-D Reconstruction Curriculum for SquatSpot

## Purpose

This curriculum focuses on the stages that precede pose-derived measurements:
converting calibrated RGB-D observations into point clouds, processing those
point clouds, registering multiple views, and reconstructing surfaces.

The goal is to learn the underlying geometry rather than treating Open3D as a
black box. Each stage will connect back to the constraints and possible future
features of SquatSpot.

## Reconstruction Pipeline

```text
RGB image + depth image
          |
          v
Validate, align, and calibrate
          |
          v
Back-project pixels into 3D
          |
          v
Single-frame colored point cloud
          |
          v
Filter noise and downsample
          |
          v
Estimate surface normals
          |
          v
Align multiple point clouds
          |
          v
Fuse observations into a TSDF volume
          |
          v
Extract and optionally process a mesh
```

## Stage 1: Validate and Prepare RGB-D Data

Before creating a point cloud, we must understand and validate the sensor data:

- Confirm that RGB and depth frames are synchronized.
- Determine whether depth is already aligned with the RGB camera.
- Identify invalid or missing depth values.
- Determine the depth units and scale.
- Apply an appropriate minimum and maximum depth range.
- Obtain the camera intrinsics.
- Account for distortion or use already-undistorted images.
- Optionally filter noise in the depth-image domain.

The camera intrinsic matrix is:

```text
K = [[fx,  0, cx],
     [ 0, fy, cy],
     [ 0,  0,  1]]
```

The parameters `fx` and `fy` describe focal length in pixels. The parameters
`cx` and `cy` describe the principal point.

An RGB image and depth image are not sufficient for metrically correct
back-projection unless their alignment, depth encoding, and camera calibration
are known.

## Stage 2: Back-Project Depth Pixels

For a valid depth pixel at image coordinates `(u, v)`:

```text
Z = depth(u, v) / depth_scale
X = (u - cx) * Z / fx
Y = (v - cy) * Z / fy
```

The resulting point is:

```text
p_camera = [X, Y, Z]
```

The corresponding RGB pixel supplies the point's color.

This stage converts a two-dimensional image grid into a set of three-dimensional
points in the camera coordinate system. Open3D provides convenience functions
for this operation, but we will first understand and test the equations
directly.

## Stage 3: Clean and Downsample the Point Cloud

Once the point cloud exists, common processing operations include:

### Voxel downsampling

Space is divided into cubic voxels. Points within a voxel are replaced by a
representative point.

Voxel downsampling:

- Reduces point count.
- Reduces memory and processing cost.
- Creates a more uniform spatial density.
- Can reduce visible noise through local averaging.
- May remove important detail if the voxel size is too large.

Voxel downsampling is primarily a density and computational-control operation.
It is not inherently a surface-smoothing algorithm.

### Statistical outlier removal

Points are removed when their average neighbor distance is unusually large
relative to the rest of the cloud.

This is useful for isolated sensor noise, but aggressive thresholds may remove
valid thin or distant structures.

### Radius outlier removal

A point is removed if it does not have enough neighbors within a chosen radius.

The radius should reflect the point-cloud scale and density. Parameters suitable
for a cloud expressed in meters will not be suitable for the same cloud
expressed in millimeters.

### Cropping

Spatial cropping can remove irrelevant regions such as distant walls, the
ceiling, or areas outside the workout space.

## Stage 4: Estimate Surface Normals

A point contains a position but does not inherently contain a surface
orientation. Open3D can estimate a normal from neighboring points.

Normals are useful for:

- Point-to-plane ICP.
- Surface analysis.
- Lighting and visualization.
- Surface and mesh reconstruction.

Normal-estimation neighborhoods should usually be chosen relative to the voxel
size and local point density. Normals may also require consistent orientation.

## Stage 5: Register Multiple Views

Point clouds produced by different camera poses initially occupy different
coordinate systems. Registration estimates transformations that place them in a
common world coordinate system:

```text
p_world = T_camera_to_world * p_camera
```

A registration workflow may include:

- A known initial transformation.
- Feature-based coarse registration.
- RGB-D odometry.
- Point-to-point ICP.
- Point-to-plane ICP.
- Pose-graph optimization for longer sequences.

Registration must be evaluated quantitatively and visually. A plausible-looking
result can still be geometrically incorrect.

## Stage 6: Fuse Observations with a TSDF

Simply concatenating registered point clouds often leaves noise, duplicated
surfaces, and inconsistent density. TSDF fusion integrates repeated depth
observations into a voxel volume containing a truncated signed distance field.

TSDF fusion can:

- Combine repeated surface observations.
- Reduce some sensor noise.
- Represent a coherent implicit surface.
- Associate color with reconstructed geometry.
- Support extraction of a triangle mesh.

Important parameters include:

- Voxel size or volume resolution.
- SDF truncation distance.
- Depth scale.
- Depth truncation distance.
- Camera intrinsics.
- The camera pose for every integrated frame.

## Stage 7: Extract and Process a Mesh

After TSDF integration, the implicit surface can be converted into a triangle
mesh. Post-processing operations may include:

- Computing vertex normals.
- Removing small disconnected components.
- Laplacian smoothing.
- Taubin smoothing.
- Vertex clustering.
- Quadric decimation.

Smoothing is not automatically beneficial. Excessive smoothing can remove
geometric evidence and distort measurements. Mesh processing should have a
specific objective and should be evaluated against the unsmoothed reconstruction.

## SquatSpot-Specific Constraints

Standard RGB-D fusion assumes that the observed scene is static. A workout
recording contains both static and dynamic geometry:

- The camera may be stationary.
- The room and floor are stationary.
- The athlete is moving.

Integrating a moving athlete into one static TSDF volume would produce ghosted
or duplicated body surfaces.

For SquatSpot, the initial product-relevant approach should therefore be:

1. Construct a point cloud for an individual RGB-D frame.
2. Segment or crop the athlete from the surrounding scene.
3. Analyze or visualize the athlete per frame.
4. Avoid fusing the moving body across time with ordinary static-scene TSDF.

Possible later approaches include dynamic reconstruction, non-rigid
registration, body-model fitting, or phase-specific reconstruction. These are
more advanced than standard static RGB-D fusion.

## Relationship to the Local MM-Fit Data

The local `assets/data/mm-fit` copy contains pose estimates, wearable signals,
and exercise labels, but not RGB or depth videos.

The separately published MM-Fit RGB-D data was captured at 30 Hz with an Orbbec
Astra Pro. The complete video release is large; for example, the `w00` RGB video
is approximately 2.2 GB and its depth video is approximately 108 MB.

Before using those videos for back-projection, we would still need to establish:

- How depth values are encoded in the compressed video.
- The depth scale and invalid-depth representation.
- The camera intrinsics.
- Whether RGB and depth frames are spatially registered.
- Whether video compression preserved sufficient depth precision.

We will therefore begin with a known-good, calibrated Open3D RGB-D sample before
attempting to decode the MM-Fit videos.

## Practical Learning Sequence

### Lesson 1: Construct one calibrated RGB-D point cloud

- Inspect one RGB frame.
- Inspect its corresponding depth frame.
- Examine image sizes and data types.
- Identify invalid depth pixels.
- Interpret the intrinsic matrix.
- Understand `depth_scale` and `depth_trunc`.
- Create and visualize a colored point cloud.

### Lesson 2: Reproduce back-projection manually

- Select a small number of depth pixels.
- Calculate their XYZ coordinates using the pinhole-camera equations.
- Compare the manual results with Open3D's generated points.
- Investigate image-coordinate and camera-coordinate conventions.

### Lesson 3: Downsample and filter

- Apply several voxel sizes.
- Record the resulting point counts.
- Compare preserved detail and computational cost.
- Apply radius and statistical outlier removal.
- Determine which points are removed and why.

### Lesson 4: Estimate and inspect normals

- Estimate normals with multiple neighborhood sizes.
- Visualize their orientations.
- Identify unstable normals near boundaries and noisy regions.
- Relate the search radius to voxel size.

### Lesson 5: Register two static views

- Construct point clouds from two RGB-D frames.
- Apply an initial pose estimate.
- Refine alignment with ICP.
- Compare point-to-point and point-to-plane ICP.
- Interpret fitness and inlier RMSE.

### Lesson 6: Register a sequence

- Estimate consecutive camera transformations.
- Accumulate poses.
- Observe drift.
- Construct and optimize a pose graph.
- Identify loop-closure opportunities.

### Lesson 7: TSDF fusion

- Create a scalable or uniform TSDF volume.
- Integrate calibrated RGB-D frames using their camera poses.
- Vary voxel and truncation parameters.
- Extract and inspect a triangle mesh.

### Lesson 8: Mesh processing

- Remove disconnected components.
- Compute normals.
- Compare unsmoothed, Laplacian-smoothed, and Taubin-smoothed results.
- Decimate the mesh and evaluate lost detail.

### Lesson 9: Dynamic-scene failure analysis

- Attempt static fusion on frames containing motion.
- Identify ghosting and duplicated geometry.
- Explain which reconstruction assumptions were violated.
- Separate camera motion from subject motion.

### Lesson 10: SquatSpot application

- Create a per-frame point cloud of an athlete.
- Segment or crop the athlete.
- Overlay or compare pose landmarks with observed depth geometry.
- Explore depth-supported measurements and confidence checks.
- Decide whether RGB-D hardware is appropriate for a future kiosk or clinical
  version of SquatSpot.

## Immediate First Exercise

Start with one known-good Open3D RGB-D sample containing:

- One color image.
- One aligned depth image.
- A documented depth scale.
- A calibrated intrinsic matrix.

Before creating the point cloud, answer:

1. What are the RGB and depth image dimensions?
2. What are their data types?
3. How many depth pixels are valid?
4. What physical value does one depth unit represent?
5. Which coordinate system will contain the generated points?
6. What effect should increasing `depth_scale` have?
7. What effect should reducing `depth_trunc` have?
8. What is the maximum possible point count before filtering?

After answering these questions, create the colored point cloud and compare its
raw and voxel-downsampled forms.

