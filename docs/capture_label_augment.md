# Capture, Web Labeling, and Augmentation Workflow

This workflow creates the first stator detection dataset from one CSI camera.
Start with `sensor_id=0` and one class: `stator`.

## Web workflow

Launch the Web workspace:

```bash
python3 run_web.py --host 0.0.0.0 --port 8000
```

Use the workflow in order:

1. Capture images into `data/frames/raw`.
2. Open `http://127.0.0.1:8000`, draw tight boxes around each stator, and save.
3. Open “数据与训练”, validate labels, split the dataset, and optionally augment the training split.
4. Train the model and test the current image with the generated `best.pt`.
5. Export `best.pt` to TensorRT only when deploying to the target device.

The workflow uses these project directories:

- sampled frames: `data/frames/raw/`
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

## 2. Label the stator

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

## 3. Validate labels

```bash
python3 scripts/check_yolo_labels.py \
  --images-dir data/labeling/export/images \
  --labels-dir data/labeling/export/labels
```

Fix all missing labels, malformed rows, or out-of-range coordinates before training.

## 4. Split the dataset

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

## 5. Augment the training split

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

## 6. First acceptance check

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
