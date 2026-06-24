# Stator YOLO Bundle

This folder is a self-contained project bundle for the stator YOLO workflow.

## Start

```bash
python3 run_gui.py
```

Environment check:

```bash
python3 run_env_check.py
```

Optional editable install:

```bash
python3 -m pip install -e .
stator-yolo gui
stator-yolo env
```

## Notes

- The GUI can start without a connected camera.
- RealSense camera functions require `pyrealsense2` and a compatible librealsense installation.
- YOLO training/inference requires the Python dependencies in `requirements.txt`.
- TensorRT `.engine` export must be done on the target Jetson/TensorRT machine.

