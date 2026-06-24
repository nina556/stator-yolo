# Stator Detection Workflow

## 1. Sampling policy

Record videos by session, not by random clips.

Each session should emphasize one dominant condition:

- lighting condition
- table or tray background
- stator pose family
- occlusion level
- robot motion state

Track session metadata in `data/manifests/session_template.csv`.

Recommended first collection target:

- 10 to 20 sessions
- 30 to 90 seconds per session
- at least 3 different scene families

## 2. Frame extraction policy

Do not annotate every frame.

Use:

- 1 to 3 FPS for static or slow scenes
- 3 to 5 FPS for active manipulation
- extra extraction around failure cases

Target the first dataset at roughly 1000 to 3000 labeled images.

## 3. Annotation rules

First version class list:

- `stator`

Boxing rules:

- keep boxes tight to the visible stator boundary
- label partially occluded stators when still recognizable
- skip frames where motion blur makes the object ambiguous
- do not label reflections that are not real objects

## 4. Split policy

Prefer splitting by session instead of random adjacent frames.

Avoid putting near-duplicate neighboring frames in both train and validation.

## 5. Augmentation policy

Useful augmentations:

- brightness and contrast
- mild rotation
- blur
- noise
- limited occlusion

Avoid unrealistic transforms:

- vertical flips
- heavy perspective distortion
- strong color changes that do not match the workcell

## 6. Deployment policy

Training stays off-board.

Jetson should only:

- load a TensorRT engine
- run inference on image or video streams
- publish boxes, labels, and confidence

Run validation on full videos before integrating into robot runtime.
