# CSI Capture, Labeling, and Augmentation Workflow

This workflow creates the first stator detection dataset from one CSI camera.
Start with `sensor_id=0` and one class: `stator`.

## GUI workflow

Launch the local GUI:

```bash
python3 scripts/stator_dataset_gui.py
```

Use the tabs in order:

1. Capture: set `sensor_id`, resolution, FPS, duration, and sample FPS, then start and stop capture.
2. Label: load sampled frames, draw tight boxes around each `stator`, then save labels.
3. Dataset: validate YOLO labels, split the dataset, then augment only the training split.
4. Train: fine-tune an official YOLO pretrained model, watch live loss and metric curves, then export `best.pt` to TensorRT `.engine`.
5. Test: load `.engine` or `.pt` and run realtime CSI camera detection.

The GUI writes the same project directories used by the command-line workflow:

- sampled frames: `data/frames/raw/<session_id>/`
- raw videos: `data/raw_videos/`
- YOLO export: `data/labeling/export/images` and `data/labeling/export/labels`
- training dataset: `data/dataset/`

## 1. Capture one session

Default capture mode is `1280x720@60` with `flip-method=2`.
Frames are sampled for labeling while the full video is also recorded.

```bash
python3 scripts/capture_csi_session.py \
  --sensor-id 0 \
  --duration-sec 60 \
  --sample-fps 2 \
  --scene first_stator_test \
  --lighting normal \
  --background workbench \
  --pose-group mixed \
  --occlusion-level low \
  --robot-state static
```

Outputs:

- sampled frames: `data/frames/raw/<session_id>/`
- raw video: `data/raw_videos/<session_id>.mp4`
- frame manifest: `data/manifests/frame_manifest.csv`
- session manifest: `data/manifests/session_manifest.csv`

Use 1080p only at 30 FPS:

```bash
python3 scripts/capture_csi_session.py \
  --sensor-id 0 \
  --width 1920 \
  --height 1080 \
  --fps 30 \
  --duration-sec 60 \
  --sample-fps 2
```

## 2. Prepare images for labeling

Create a flat image bundle for CVAT or Label Studio.

```bash
python3 scripts/prepare_labeling_bundle.py \
  --frames-dir data/frames/raw \
  --output-dir data/labeling/bundle
```

## 3. Label the stator

Create one detection class:

```text
0: stator
```

Boxing policy:

- keep the box tight to the visible stator boundary
- label partial occlusions when the stator is still recognizable
- skip ambiguous motion blur
- do not label reflections

Export labels in YOLO detection format:

```text
data/labeling/export/
├── images/
└── labels/
```

Each label line must be:

```text
class_id x_center y_center width height
```

Coordinates are normalized to `[0, 1]`.

## 4. Validate labels

```bash
python3 scripts/check_yolo_labels.py \
  --images-dir data/labeling/export/images \
  --labels-dir data/labeling/export/labels
```

Fix all missing labels, malformed rows, or out-of-range coordinates before training.

## 5. Split the dataset

For the first local smoke test, this random split is acceptable:

```bash
python3 scripts/split_dataset.py \
  --images-dir data/labeling/export/images \
  --labels-dir data/labeling/export/labels \
  --output-dir data/dataset \
  --train-ratio 0.8 \
  --val-ratio 0.1 \
  --test-ratio 0.1
```

For real model evaluation, split by session so neighboring frames do not leak
between train and validation.

## 6. Augment the training split

If `albumentations` is installed, the script uses bbox-aware rotation and blur
augmentation. Otherwise it falls back to OpenCV image-only augmentation that
keeps YOLO boxes unchanged.

```bash
python3 scripts/augment_dataset.py \
  --dataset-dir data/dataset \
  --copies-per-image 2
```

Only `images/train` and `labels/train` are augmented. Validation and test data
must stay unmodified.

## 7. First acceptance check

Before training, confirm:

- `data/dataset/images/train` has labeled images
- `data/dataset/images/val` has labeled images
- `data/dataset.yaml` points to `./data/dataset`
- class `0` is still `stator`

Then run training:

```bash
bash train/train_yolov8.sh
```

Review:

- `runs/stator_yolov8/results.csv`
- `runs/stator_yolov8/results.png`
- `runs/stator_yolov8/weights/best.pt`
