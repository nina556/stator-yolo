# 定子 YOLO 视觉识别工作区

本项目用于在 Jetson 上完成“定子识别”模型的完整工程流程：

1. 从相机采集 RGB 图像或视频。
2. 抽帧生成标注样本。
3. 标注 `stator` 检测框。
4. 校验、切分、增强数据集。
5. 基于官方 YOLO 预训练模型微调。
6. 监控训练日志和 loss/mAP 曲线。
7. 将 `best.pt` 导出为 TensorRT `.engine`。
8. 在相机视频流上实时检测定子。

当前相机方案已从 Jetson CSI 相机切换到 **深视 3D 相机**。YOLO 检测本身只依赖 RGB 图像；深度图、点云和相机内参可以作为后续定位/机械臂抓取扩展，不直接参与当前 YOLO 训练。

## 统一入口

推荐使用包入口，不要直接记一堆脚本：

浏览器标注界面（本机训练环境不是必需的）：

```bash
python3 run_web.py
```

然后访问 `http://127.0.0.1:8000`。页面支持导入图片、框选定子并保存
YOLO 标签到 `data/labeling/export/labels`。

```bash
python3 -m stator_yolo.cli gui
```

环境检查：

```bash
python3 -m stator_yolo.cli env
```

直接打开完整工作流 GUI：

```bash
python3 -m stator_yolo.gui
```

如果你还没安装成包，也可以继续用 `scripts/stator_dataset_gui.py`，两者调用的是同一套逻辑。

可选：安装为 editable 包，之后可以直接使用命令行入口：

```bash
python3 -m pip install -e .
stator-yolo gui
stator-yolo env
```

## 当前状态

- 训练、标注、增强、TensorRT 导出流程已经可用。
- GUI 已包含 `Capture / Label / Dataset / Train / Test` 页签。
- GUI 的 `Capture` 和 `Test` 页均支持 `RealSense / CSI` 切换。
- RealSense 采集会保存 RGB 彩色帧，并同步保存 16-bit 深度帧。
- 深度图用于从 2D 检测框反查 3D 坐标，后续接机械臂手眼标定。
- 没有连接摄像头时，GUI 仍可启动；相机相关按钮会在运行时给出错误提示，但不会阻止标注、数据集和训练页签使用。

## 深视 3D 相机接口测试

先用探测脚本确认相机在当前 Jetson 上暴露的接口：

```bash
python3 scripts/test_deepvision_camera.py --list-only
```

当前机器实测识别为 `Intel RealSense Depth Camera 455`，接口关系如下：

```text
/dev/video8  彩色图，YUYV，OpenCV 可直接读取
/dev/video2  深度图，Z16 16-bit depth，建议用 raw V4L2 或 RealSense SDK 读取
/dev/video6  灰度/红外图，GREY/UYVY
/dev/video3、/dev/video7、/dev/video9  元数据类节点，通常不作为图像输入
```

测试彩色图：

```bash
python3 scripts/test_deepvision_camera.py \
  --devices /dev/video8 \
  --width 640 \
  --height 480 \
  --fps 30 \
  --frames 30 \
  --fourcc YUYV
```

测试深度图：

```bash
python3 scripts/test_deepvision_camera.py \
  --devices /dev/video2 \
  --width 640 \
  --height 480 \
  --frames 5 \
  --fourcc Z16 \
  --raw-v4l2
```

输出会保存在：

```text
runs/deepvision_probe/
├── video8_frame.png             # 彩色样张
├── video2_z16.raw               # 原始 Z16 深度流
├── video2_depth_z16.npy         # 第一帧深度矩阵，uint16
├── video2_depth_z16.png         # 16-bit 深度 PNG
└── video2_depth_preview.png     # 深度伪彩预览
```

注意：当前系统没有安装 `pyrealsense2`、`pyorbbecsdk`、`openni`、`depthai` 等 3D 相机 SDK 的 Python 模块。第一版可以用 `/dev/video8` 采集 RGB 做 YOLO；如果要做 RGB-D 同步和相机内参/深度尺度读取，建议后续接入 RealSense SDK 或厂商 SDK。

如果你不接相机，也可以先把项目当作离线标注/训练工作流来用：

```bash
python3 -m stator_yolo.gui
```

Capture/Test 页会在没有相机时显示错误日志，但 Label/Dataset/Train 仍可正常工作。

## RealSense SDK 实时 RGB-D 流

由于当前相机被系统识别为 `Intel RealSense Depth Camera 455`，推荐使用 RealSense SDK 做正式 RGB-D 采集。SDK 相比 V4L2 的优势是：

- 同步获取 color/depth frame。
- 可把 depth 对齐到 color。
- 可读取 depth scale、相机内参和设备信息。
- 后续可以从 YOLO 检测框中心点反查深度，再计算 3D 坐标。

本项目提供 SDK 版测试脚本：

```bash
python3 scripts/test_realsense_sdk_stream.py \
  --color-width 640 \
  --color-height 480 \
  --color-fps 30 \
  --depth-width 640 \
  --depth-height 480 \
  --depth-fps 30 \
  --save-every 30
```

如果当前没有图形界面预览，可以关闭窗口显示：

```bash
python3 scripts/test_realsense_sdk_stream.py \
  --color-width 640 \
  --color-height 480 \
  --color-fps 30 \
  --depth-width 640 \
  --depth-height 480 \
  --depth-fps 30 \
  --frames 300 \
  --save-every 30 \
  --no-preview
```

查看 SDK 实际支持的分辨率/FPS：

```bash
python3 scripts/test_realsense_sdk_stream.py --list-profiles
```

当前机器实测 SDK 状态：

```text
pyrealsense2: 2.58.2
device: Intel RealSense D455
firmware: 5.15.1
usb_type: 2.1
depth_scale_m_per_unit: 0.00100000
```

当前稳定可用组合：

```bash
python3 scripts/test_realsense_sdk_stream.py \
  --color-width 640 \
  --color-height 480 \
  --color-fps 30 \
  --depth-width 640 \
  --depth-height 480 \
  --depth-fps 30 \
  --frames 300 \
  --save-every 30 \
  --no-preview
```

低分辨率 60FPS 组合可以启动：

```bash
python3 scripts/test_realsense_sdk_stream.py \
  --color-width 424 \
  --color-height 240 \
  --color-fps 60 \
  --depth-width 480 \
  --depth-height 270 \
  --depth-fps 60 \
  --frames 300 \
  --save-every 30 \
  --no-preview
```

不要直接使用：

```bash
--width 1920 --height 1080 --fps 60
```

当前 D455 SDK profile 没有暴露 `1920x1080@60`，同时 depth 也不支持这个分辨率/FPS。`1280x720` 彩色 profile 虽然可枚举，但当前 `usb_type: 2.1` 状态下 RGB-D 同步取流会出现等不到帧；如果必须使用更高分辨率，应先确认相机连接到 USB3 口并使用合格 USB3 数据线。

如果画面方向和安装方向不一致，可以加：

```bash
--flip vertical
```

可选值：

```text
none / vertical / horizontal / both
```

输出目录：

```text
runs/realsense_sdk_probe/
├── frame_000000_color.jpg
├── frame_000000_depth_z16.png
├── frame_000000_depth_preview.png
└── manifest.csv
```

`manifest.csv` 会记录帧号、SDK 时间戳、RGB 路径、深度路径、中心点深度、有效最小/最大深度。

如果 SDK Python binding 未安装，脚本会先报：

```text
Missing dependency: pyrealsense2.
```

需要先安装和当前 JetPack/TensorRT 环境匹配的 `librealsense` 与 Python binding。Jetson 上不建议随意安装不匹配的 pip wheel，优先使用 Intel/相机厂商提供的 Jetson 兼容安装方式。

## RealSense RGB-D 双视图 GUI

实时查看 2D 彩色图和 3D 深度视角：

```bash
python3 scripts/realsense_rgbd_viewer_gui.py \
  --color-width 640 \
  --color-height 480 \
  --color-fps 30 \
  --depth-width 640 \
  --depth-height 480 \
  --depth-fps 30
```

GUI 左侧显示 RGB 彩色流，右侧默认显示由深度图和相机内参生成的 3D 点云视角。右侧也可以切换为深度伪彩图。

GUI 控件：

- `Start / Stop`：启动或停止 RealSense RGB-D 流。
- `Save Snapshot`：保存当前 RGB、16-bit 深度图和深度预览图到 `runs/realsense_rgbd_viewer/`。
- `Right View`：切换右侧 `3d` 或 `depth`。
- `Flip`：按安装方向翻转 RGB 和深度帧。
- `Point Step`：3D 点云采样间隔，数值越小点越密，但越吃 CPU。
- `Min Depth / Max Depth`：过滤深度范围。
- `Yaw / Pitch`：调整右侧 3D 视角。

如果相机画面倒置：

```bash
python3 scripts/realsense_rgbd_viewer_gui.py \
  --color-width 640 \
  --color-height 480 \
  --color-fps 30 \
  --depth-width 640 \
  --depth-height 480 \
  --depth-fps 30 \
  --flip vertical
```

当前 GUI 使用 Tkinter + OpenCV + RealSense SDK，不依赖 Open3D。3D 视角是从深度图实时采样并投影出来的轻量点云预览，适合调试安装视角、深度范围和后续定子定位流程。

## RealSense 接入 YOLO 实时识别

YOLO 仍然跑 2D RGB 图像；RealSense 深度图用于把检测框中心点转换为相机坐标系下的 3D 点。

完整工作流 GUI：

```bash
python3 scripts/stator_dataset_gui.py
```

GUI 功能：

- 保留原有 `Capture / Label / Dataset / Train / Test` 全流程页签。
- 选择 `.pt` 或 `.engine` 模型。
- 设置 RealSense color/depth 分辨率和 FPS。
- 设置置信度、IOU、推理尺寸和 CUDA device。
- 设置深度范围、深度中值窗口和画面翻转。
- `Start Detection / Stop` 控制实时检测。
- 实时画面中显示 YOLO 框、中心点、深度和相机坐标系 `x/y/z`。
- `Test` 页默认使用 RealSense；如需旧 CSI 测试，可在 `Camera` 中切换为 `CSI`。

建议优先使用这个 GUI 做现场测试；下面的命令行脚本适合无显示环境、批量测试或调试。`scripts/realsense_yolo_gui.py` 是独立 RealSense 检测界面，只用于备用调试，不包含标注、数据集和训练页签。

实时预览：

```bash
python3 scripts/infer_realsense_rgbd.py \
  --model runs/stator_yolov8/weights/best.pt \
  --color-width 640 \
  --color-height 480 \
  --color-fps 30 \
  --depth-width 640 \
  --depth-height 480 \
  --depth-fps 30 \
  --conf 0.25 \
  --device 0
```

无预览测试并保存结果：

```bash
python3 scripts/infer_realsense_rgbd.py \
  --model runs/stator_yolov8/weights/best.pt \
  --color-width 640 \
  --color-height 480 \
  --color-fps 30 \
  --depth-width 640 \
  --depth-height 480 \
  --depth-fps 30 \
  --frames 300 \
  --save-csv \
  --save-video \
  --no-preview \
  --device 0 \
  --output-dir runs/realsense_yolo
```

输出：

```text
runs/realsense_yolo/
├── detections.csv
└── realsense_yolo.mp4
```

`detections.csv` 字段：

```text
frame_index,timestamp_ms,class_id,class_name,confidence,
x1,y1,x2,y2,center_x,center_y,
depth_m,camera_x_m,camera_y_m,camera_z_m
```

其中：

- `x1,y1,x2,y2` 是 YOLO 在 RGB 图上的检测框。
- `center_x,center_y` 是检测框中心像素。
- `depth_m` 是中心点附近窗口的中位深度。
- `camera_x_m,camera_y_m,camera_z_m` 是 SDK 内参反投影得到的相机坐标系 3D 点。

机械臂抓取时，下一步需要把：

```text
camera_x_m,camera_y_m,camera_z_m
```

通过手眼标定外参转换到机械臂 `base_link` 或机器人基坐标系。

## 深视 3D 相机数据约定

建议每次采集以 session 为单位保存：

```text
data/
├── raw_videos/
│   └── <session_id>_color.mp4
├── frames/raw/
│   └── <session_id>/
│       ├── <session_id>_f000000.jpg
│       ├── <session_id>_f000001.jpg
│       └── ...
├── depth/raw/
│   └── <session_id>/
│       ├── <session_id>_f000000.png
│       ├── <session_id>_f000001.png
│       └── ...
└── manifests/
    ├── frame_manifest.csv
    └── session_manifest.csv
```

RGB 图像用于 YOLO 训练。深度图使用 16-bit PNG 或 SDK 推荐格式保存，保持和 RGB 帧同名同序号，方便后续对齐。

`frame_manifest.csv` 建议字段：

```text
session_id,camera_type,frame_index,timestamp_sec,color_path,depth_path,width,height
```

`session_manifest.csv` 建议字段：

```text
session_id,camera_type,scene,lighting,background,pose_group,occlusion_level,robot_state,notes
```

## 目录结构

```text
yolo/
├── data/
│   ├── raw_videos/              # 原始 RGB 视频
│   ├── frames/raw/              # 抽帧后的 RGB 图像
│   ├── depth/raw/               # 深视 3D 深度帧，后续定位用
│   ├── labeling/export/images/  # 已标注 RGB 图片
│   ├── labeling/export/labels/  # YOLO txt 标签
│   ├── dataset/                 # train/val/test 数据集
│   ├── dataset.yaml             # YOLO 数据集配置
│   └── manifests/               # 采集元数据
├── docs/
├── export/
├── scripts/
├── train/
└── runs/                        # 训练和导出结果
```

## 推荐主流程

### 1. 采集深视 3D 相机数据

使用深视 3D 相机 SDK 或厂商示例程序采集 RGB 视频/帧。当前项目要求至少输出 RGB 图像到：

```text
data/frames/raw/<session_id>/
```

如果 SDK 支持同步深度图，建议同时输出到：

```text
data/depth/raw/<session_id>/
```

采集建议：

- 第一版先只训练 `stator` 一个类别。
- 每个 session 只强调一种主要工况，例如光照、背景、姿态、遮挡或机械臂状态。
- 静态场景每秒抽 1-3 帧。
- 操作过程每秒抽 3-5 帧。
- 第一版目标是 1000-3000 张已标注 RGB 图像。

### 2. 准备标注图片

如果 RGB 帧已经在 `data/frames/raw/`，可以直接使用 GUI 标注。也可以打包成平铺目录：

```bash
python3 scripts/prepare_labeling_bundle.py \
  --frames-dir data/frames/raw \
  --output-dir data/labeling/bundle
```

### 3. 标注定子

类别固定为：

```text
0: stator
```

标注规则：

- 检测框紧贴可见定子边界。
- 定子被部分遮挡但仍可识别时需要标注。
- 运动模糊严重、无法判断边界的图片跳过。
- 不标注反光、影子或非真实定子。

YOLO 标签格式：

```text
class_id x_center y_center width height
```

坐标均归一化到 `[0, 1]`。

标注导出目录：

```text
data/labeling/export/
├── images/
└── labels/
```

### 4. 校验并切分数据集

```bash
python3 scripts/check_yolo_labels.py \
  --images-dir data/labeling/export/images \
  --labels-dir data/labeling/export/labels

python3 scripts/split_dataset.py \
  --images-dir data/labeling/export/images \
  --labels-dir data/labeling/export/labels \
  --output-dir data/dataset \
  --train-ratio 0.8 \
  --val-ratio 0.1 \
  --test-ratio 0.1
```

注意：真实评估时最好按 session 切分，不要把相邻帧同时放进 train 和 val。

### 5. 数据增强

```bash
python3 scripts/augment_dataset.py \
  --dataset-dir data/dataset \
  --copies-per-image 2
```

增强只作用于训练集：

```text
data/dataset/images/train
data/dataset/labels/train
```

如果安装了 `albumentations`，脚本会使用 bbox-aware 增强；否则自动降级为 OpenCV 图像增强，并保持 YOLO 框坐标不变。

### 6. 训练 YOLO

默认训练：

```bash
bash train/train_yolov8.sh
```

定子在画面中较小时，建议提高输入尺寸：

```bash
EPOCHS=100 IMGSZ=960 BATCH=8 DEVICE=0 \
  bash train/train_yolov8.sh yolov8n.pt data/dataset.yaml runs stator_yolov8_v2
```

显存不足时，先降低 `BATCH`，再降低 `IMGSZ`。

训练输出：

```text
runs/<run_name>/
├── results.csv
├── results.png
├── confusion_matrix.png
├── val_batch0_labels.jpg
├── val_batch0_pred.jpg
└── weights/
    ├── best.pt
    └── last.pt
```

### 7. 导出 TensorRT Engine

必须在实际运行推理的 Jetson/TensorRT 环境上导出：

```bash
bash export/export_engine.sh runs/stator_yolov8/weights/best.pt
```

等价命令：

```bash
yolo export \
  model=runs/stator_yolov8/weights/best.pt \
  format=engine \
  imgsz=640 \
  device=0 \
  half=True \
  simplify=True
```

输出：

```text
runs/stator_yolov8/weights/best.engine
```

TensorRT 在 `building FP16 engine` 阶段可能长时间没有新日志，只要 `yolo export` 进程仍在占用 CPU/GPU，就是正常构建。

### 8. 实时检测

当前 GUI 的实时检测页仍使用 CSI 相机读取逻辑。深视 3D 相机切换后，需要把 `Test` 页的相机读取部分替换为深视 SDK RGB 帧读取。

模型推理逻辑不变：

```python
from ultralytics import YOLO

model = YOLO("runs/stator_yolov8/weights/best.engine")
result = model.predict(source=color_frame, conf=0.25, verbose=False)[0]
annotated = result.plot()
```

也可以先使用录制好的 RGB 视频离线验证：

```bash
python3 scripts/infer_video.py \
  --model runs/stator_yolov8/weights/best.pt \
  --video data/raw_videos/<session_id>_color.mp4 \
  --output runs/infer/result.mp4 \
  --conf 0.25
```

## GUI 使用说明

启动 GUI：

```bash
python3 scripts/stator_dataset_gui.py
```

当前页签：

1. `Capture`
   - CSI 历史采集入口。
   - 深视 3D 相机接入后，应替换此页底层相机读取逻辑。

2. `Label`
   - 可继续使用。
   - 从 `data/frames/raw` 加载 RGB 图片并保存 YOLO 标签。

3. `Dataset`
   - 可继续使用。
   - 校验、切分、增强数据。

4. `Train`
   - 可继续使用。
   - 启动训练、查看日志和曲线、导出 TensorRT engine。

5. `Test`
   - CSI 历史实时检测入口。
   - 深视 3D 相机接入后，应替换为 SDK RGB 流。

## 深视 3D 相机接入建议

建议新增一个相机适配脚本，例如：

```text
scripts/capture_deepvision_session.py
```

职责：

- 初始化深视 3D 相机。
- 获取同步 RGB 帧和深度帧。
- RGB 帧保存为 `.jpg`。
- 深度帧保存为 16-bit `.png` 或 SDK 推荐格式。
- 写入 `frame_manifest.csv` 和 `session_manifest.csv`。
- 可选：保存 RGB 视频到 `data/raw_videos/`。

建议新增一个实时检测脚本，例如：

```text
scripts/infer_deepvision_realtime.py
```

职责：

- 从深视 3D 相机读取 RGB 帧。
- 使用 `.pt` 或 `.engine` 做 YOLO 检测。
- 可选：用检测框中心点查询深度，输出 3D 坐标。

## 训练现象排查

如果 `confusion_matrix.png` 看起来全部是 background，先检查：

```bash
python3 scripts/check_yolo_labels.py \
  --images-dir data/dataset/images/val \
  --labels-dir data/dataset/labels/val
```

再看：

```text
runs/<run_name>/val_batch0_labels.jpg
runs/<run_name>/val_batch0_pred.jpg
```

- `labels` 图有框：说明验证集标签被正确读取。
- `pred` 图无框：说明模型置信度太低或训练不足。

小数据集初期可以临时降低置信度测试：

```bash
yolo predict \
  model=runs/stator_yolov8/weights/best.pt \
  source=data/dataset/images/val \
  imgsz=640 \
  conf=0.05 \
  device=0
```

长期解决方案是补数据、提高目标占图比例、提高 `IMGSZ` 或加入 ROI 裁剪。

## CSI 旧入口

以下脚本是旧 CSI 相机流程，保留作参考或备用：

- `test_camera_csi.py`
- `scripts/capture_csi_session.py`
- GUI 中的 `Capture` 和 `Test` 相机流读取逻辑

CSI 支持模式：

- `3280x2464@21`
- `3280x1848@28`
- `1920x1080@30`
- `1640x1232@30`
- `1280x720@60`

## 关键文件

- GUI 工作流：`scripts/stator_dataset_gui.py`
- 数据校验：`scripts/check_yolo_labels.py`
- 数据切分：`scripts/split_dataset.py`
- 数据增强：`scripts/augment_dataset.py`
- 图片推理：`scripts/infer_image.py`
- 视频推理：`scripts/infer_video.py`
- 训练脚本：`train/train_yolov8.sh`
- TensorRT 导出：`export/export_engine.sh`
- 数据集配置：`data/dataset.yaml`
- 旧 CSI 预览：`test_camera_csi.py`
- 旧 CSI 采集：`scripts/capture_csi_session.py`
