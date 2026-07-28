# LUPI Training Architecture

## Goal

Train a squat-form model that benefits from synchronized, triangulated 3D skeletons
during training but requires only one front- or side-view skeleton at deployment.

The system must distinguish between:

1. The probability that a form fault is present.
2. Whether that fault is observable from the supplied view.
3. Whether the landmark sequence has sufficient quality.
4. Whether to report a result or abstain.

Privileged supervision cannot recover information that is fundamentally absent from a
single view. The deployed system should therefore return qualified feedback or
`insufficient evidence` instead of making a confident unsupported prediction.

## Recommended First Architecture

```text
                  TRAINING ONLY

       Triangulated, normalized 3D sequence
                       |
          +------------+-------------+
          |                          |
   3D teacher classifier      3D biomechanical targets
   p_T(form | X_3D)           angles, depth, valgus,
                              stance, lean, symmetry
          |                          |
          +------ privileged --------+
                 supervision
                       |
           +-----------+-----------+
           |                       |
   Front-view student       Side-view student
           |                       |
           +-----------+-----------+
                       |
          +------------+-------------+
          |            |             |
     Form scores   Biomechanical   Observability
                   estimates and    and input
                   uncertainty       quality
          |            |             |
          +------------+-------------+
                       |
          Calibration and abstention
                       |
        +--------------+---------------+
        |              |               |
     Feedback     Qualified feedback    Insufficient
                                          evidence
```

### 1. Privileged 3D teacher

Train a compact teacher on complete 3D pose sequences. Candidate implementations:

- Engineered biomechanical features with gradient-boosted trees.
- A small temporal CNN.
- A small temporal GCN.

An ensemble can provide more stable soft targets and an estimate of teacher
uncertainty. The teacher is never required at deployment.

### 2. View-specific students

Start with separate front- and side-view students because the views expose different
biomechanical evidence.

| Feature or fault | Front view | Side view |
|---|---:|---:|
| Stance width | Strong | Weak |
| Knee valgus | Strong | Weak |
| Squat depth | Moderate | Strong |
| Trunk lean | Moderate | Strong |
| Knee-over-toe motion | Weak | Strong |
| Left/right asymmetry | Strong | Weak |

Each student receives a normalized single-view landmark sequence plus landmark
visibility/confidence and estimated camera-view metadata.

### 3. Student outputs

Each student should produce:

- Form-class probabilities.
- Estimates of recoverable biomechanical measurements.
- Uncertainty for each measurement.
- A per-fault observability score.
- An overall input-quality score.

The observability output must distinguish `fault not detected` from `fault cannot be
assessed from this view`.

## Training Objectives

For a single-view input \(X_v\), its label \(y\), and the paired privileged 3D
sequence \(X_{3D}\), use:

\[
\mathcal{L} =
\lambda_{label}\mathcal{L}_{label}
+ \lambda_{KD}\mathcal{L}_{KD}
+ \lambda_{bio}\mathcal{L}_{bio}
+ \lambda_{repr}\mathcal{L}_{repr}
+ \lambda_{obs}\mathcal{L}_{obs}
\]

Where:

- **Label loss:** classification loss against the manually annotated exercise form.
- **Soft distillation loss:** match the softened class distribution from the 3D
  teacher.
- **Biomechanical loss:** predict measurements calculated from the privileged 3D
  skeleton.
- **Representation loss:** optionally align selected student and teacher embeddings.
- **Observability loss:** predict whether each fault can be assessed from the current
  view and landmark quality.

Distillation and biomechanical losses should be masked or down-weighted when a target
is not observable from the student's view. The student should not be penalized for
failing to infer genuinely hidden geometry.

## Training Data Preparation

Treat a complete repetition as one independent sample. Never split individual frames
from the same repetition between training, validation, or test sets.

For every EC3D repetition:

1. Retain the normalized triangulated 3D sequence as privileged input.
2. Construct paired front- and side-view landmark sequences.
3. Preserve subject, repetition, instruction, camera, and frame identities.
4. Calculate privileged biomechanical targets from 3D.
5. Add deployment-like corruptions to the single-view inputs:
   - landmark jitter and confidence variation;
   - missing joints and missing frame runs;
   - camera yaw, pitch, scale, and translation changes;
   - perspective variation and imperfect front/side angles;
   - cropped feet or other partial visibility.

Synthetic projections increase camera-condition diversity but do not create new
independent people or motion examples. Final validation must use landmarks produced
by the actual deployment pose pipeline.

## Inference Policy

At deployment:

1. Estimate whether the input is front, side, oblique, or unsupported.
2. Select the corresponding student or condition a shared student on view angle.
3. Check pose coverage and landmark quality.
4. Predict form, biomechanical measurements, uncertainty, and observability.
5. Calibrate the class probabilities.
6. Report a fault only when both confidence and observability exceed validated
   thresholds.
7. Otherwise provide qualified feedback or request a better camera view.

Example outputs:

- `Likely insufficient squat depth (82% calibrated confidence).`
- `Knee alignment appears acceptable from this front view.`
- `Knee alignment cannot be assessed reliably from this side view.`
- `Insufficient landmark quality—please keep both feet and hips visible.`

## Calibration and Abstention

Begin with:

- A small ensemble of independently trained students.
- Temperature scaling on a subject-held-out calibration set.
- A reporting threshold selected from an accuracy-versus-coverage curve.

Later, evaluate conformal prediction sets so the system can return alternatives such
as `{correct, not low enough}` instead of forcing a single label.

Calibration must eventually be repeated on product-domain recordings. Calibration on
EC3D alone will not protect against the domain shift to MediaPipe or another
single-camera pose estimator.

## Initial Experiment Ladder

1. Train a single-view engineered-feature baseline without privileged information.
2. Add soft-label distillation from the 3D teacher.
3. Add privileged biomechanical-feature prediction.
4. Compare one shared view-conditioned student with separate front/side experts.
5. Add landmark and camera corruption augmentation.
6. Add ensemble uncertainty and temperature scaling.
7. Add observability gating and abstention.
8. Validate on unseen subjects processed through the product landmark pipeline.

Each stage should be retained only if it improves subject-held-out results.

## Evaluation

Use subject-held-out evaluation, ideally leave-one-subject-out while EC3D remains
small. Report:

- Macro F1 and balanced accuracy.
- Per-form and per-view precision and recall.
- Negative log-likelihood and Brier score.
- Expected calibration error.
- Accuracy-versus-coverage and risk-versus-coverage.
- Rate of incorrect predictions above high-confidence thresholds.
- Observability and abstention accuracy.
- Biomechanical feature error and interval coverage.

The primary safety-oriented outcome is not maximum raw classification accuracy. It is
high accuracy on accepted predictions, a low rate of confidently incorrect feedback,
and appropriate abstention when the available view is insufficient.
