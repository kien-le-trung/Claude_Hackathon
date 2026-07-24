# Open3D and 3D Geometry Learning Plan for SquatSpot

## Purpose

This learning track connects 3D geometry and Open3D concepts directly to useful SquatSpot product features. The main product goal is a 3D squat replay with geometry-based, rep-level form analysis.

SquatSpot already stores MediaPipe's 33 `world_landmarks` for each detected pose. These provide a practical bridge into 3D geometry, but they must be interpreted correctly: they are estimated 3D body coordinates from monocular video, not depth-sensor measurements or a dense reconstruction of the athlete and surrounding scene.

## Relationship to MM-Fit

The local `assets/data/mm-fit` dataset contains 21 workout sessions with:

- 2D poses shaped as `(2, frames, 19 joints)`
- 3D poses shaped as `(3, frames, 18 joints)`
- Wearable sensor signals
- Exercise and repetition labels

This local copy does not contain RGB images, depth images, camera intrinsics, or camera trajectories. It can support learning about 3D pose, coordinate systems, temporal movement, and skeleton normalization, but it cannot directly support dense RGB-D reconstruction.

MM-Fit labels identify exercises and repetitions rather than good or bad form. Consequently, MM-Fit may help us study motion segmentation and representation, but it should not be treated as direct supervision for SquatSpot's form-quality classifier.

## Product Applications

### 1. Interactive 3D squat replay

Convert stored MediaPipe world landmarks into an animated skeleton that users can inspect from the front, side, or another angle.

Open3D can be used during development to prototype and validate the geometry. A browser-native renderer would likely be more appropriate for the final interactive user interface.

### 2. Explainable form measurements

Derive interpretable measurements from 3D landmarks, including:

- Knee flexion and squat depth
- Torso lean
- Hip-knee-ankle alignment
- Heel movement
- Stance width
- Left-right asymmetry
- Center-of-body trajectory
- Joint velocity and stability

These measurements can complement the existing classifier and produce feedback tied to repetitions and movement phases. For example:

> Excessive forward torso lean appeared near the bottom of 4 of 6 repetitions.

This is more actionable and auditable than returning only a frame-level class such as `bad_back`.

### 3. Repetition and phase detection

Use hip trajectory, knee angles, and other temporal signals to divide a squat into phases:

`standing -> descent -> bottom -> ascent -> standing`

This would allow SquatSpot to:

- Count repetitions
- Score repetitions separately
- Identify when a form error occurs
- Select informative replay frames
- Avoid treating adjacent video frames as independent examples

Sequence-aware evaluation would also help address the current model warning about leakage from adjacent poses across training and test data.

### 4. Pose normalization

Express each skeleton in a body-centered coordinate system. A possible convention is:

- Origin at the midpoint of the hips
- Vertical axis aligned with the body or an estimated gravity direction
- Horizontal axis aligned across the hips
- Distances normalized by stable body proportions

This applies coordinate-frame and transformation concepts from 3D reconstruction. It may make analysis less sensitive to camera position, video resolution, athlete size, and distance from the camera.

### 5. Reference-motion comparison

Compare a user's normalized squat trajectory with a reference trajectory. The comparison should consider complete movement phases rather than isolated frames and should account for differences in body proportions and execution speed.

## Dense RGB-D Reconstruction

Dense RGB-D reconstruction is educational but is not currently an MVP priority for SquatSpot:

- The current product accepts ordinary videos without a depth channel.
- A moving athlete violates the static-scene assumption used by standard TSDF reconstruction.
- Most immediately useful squat feedback can be derived from pose landmarks.
- Requiring depth-camera hardware would narrow the accessible product audience.

RGB-D reconstruction could become valuable in a future gym kiosk, coaching station, or physical-therapy version of SquatSpot. A fixed depth camera could provide metric joint positions, floor geometry, and stronger evidence about foot contact.

## Instructor-Led Learning Sequence

The work will follow an instructor-guided approach. The learner writes the implementation, while the instructor provides concepts, assignments, review, and progressively stronger hints rather than complete code.

### Lesson 1: 3D coordinate systems

Visualize and compare an MM-Fit 3D pose and a SquatSpot MediaPipe pose. Determine their skeleton definitions, array layouts, axes, origins, handedness, and likely units.

Product connection: establish a reliable representation for SquatSpot's 3D replay.

### Lesson 2: Rigid transformations and normalization

Translate, rotate, and scale skeletons into comparable body-centered coordinate systems.

Product connection: reduce sensitivity to athlete size and camera viewpoint.

### Lesson 3: 3D kinematics

Measure joint angles, squat depth, torso lean, stance width, symmetry, and landmark trajectories.

Product connection: generate interpretable evidence for form feedback.

### Lesson 4: Temporal geometry

Analyze full motion sequences and detect repetitions and squat phases.

Product connection: replace isolated frame judgments with rep-level analysis.

### Lesson 5: Open3D diagnostic replay

Create an offline visualization for inspecting skeletons, coordinate axes, low-confidence landmarks, motion trails, and classifier decisions.

Product connection: give developers a tool for debugging extraction and classification behavior before exposing a simplified replay to users.

### Lesson 6: Product capstone

Design a 3D Form Replay result containing:

- A rotatable skeleton
- A repetition timeline
- Highlighted joints or segments associated with detected problems
- Measurements supporting each conclusion
- Plain-language feedback

### Lesson 7: Optional RGB-D extension

After the landmark-based feature works, use a separate RGB-D dataset to learn:

- Depth back-projection
- Colored point-cloud construction
- Camera intrinsics and extrinsics
- Pairwise registration and ICP
- Camera trajectories
- TSDF integration and mesh extraction

The results of this lesson should be treated as an exploration of a future depth-camera product rather than a requirement for the current upload workflow.

## Recommended First Milestone

Take one completed SquatSpot upload and determine how its stored MediaPipe world landmarks should become a normalized 3D skeleton and rep-level analysis.

The milestone should answer these questions before implementation begins:

1. Which MediaPipe landmarks are required for squat analysis?
2. Which coordinate frame should SquatSpot use internally?
3. How should the skeleton be normalized without removing meaningful form differences?
4. Which trajectory best identifies squat repetitions and phases?
5. Which measurements can support the current `good`, `bad_back`, and `bad_heel` outputs?
6. How should low landmark visibility or missing frames affect the confidence of the final feedback?

This milestone teaches practical 3D geometry while producing work that can directly improve SquatSpot.
