# 定子 YOLO Web 工作台

这是一个面向定子目标检测的数据标注、数据集处理、模型训练与部署项目。

项目现在以浏览器工作台为主界面。局域网用户可以通过浏览器上传图片、框选定子、保存 YOLO 标签、处理数据集并启动 GPU 训练，不需要在每台电脑上安装 Python 或训练环境。

## 当前功能

网页工作台已经支持：

- 批量上传 JPG、PNG、WEBP、BMP 图片。
- 浏览和切换待标注图片。
- 鼠标拖拽绘制定子检测框。
- 撤销、清空和保存 YOLO 标签。
- 保存标签并自动切换到下一张。
- 删除源图片，并同步清理对应导出图片和标签。
- 显示标注进度。
- 检查图片和 YOLO 标签是否匹配。
- 按 `8:1:1` 划分训练集、验证集和测试集。
- 对训练集执行数据增强。
- 从网页启动和停止 YOLOv8n 训练。
- 查看数据集数量、任务状态和实时训练日志。

项目仍保留 RealSense、CSI 相机采集和推理脚本作为底层工具。旧桌面 GUI 不再是默认入口，后续功能以 Web 工作台为准。

## 系统环境

当前开发环境：

- Windows 11 + WSL2
- Ubuntu 24.04
- Python 3.12
- NVIDIA GeForce RTX 5060 Ti 8 GB
- PyTorch 2.13 + CUDA 13
- Ultralytics YOLO

建议配置：

- NVIDIA 显卡，显存 6 GB 以上
- 内存 16 GB 左右
- Python 3.10 以上
- 20 GB 以上可用磁盘空间

没有 NVIDIA GPU 也能使用图片标注和数据集处理功能，但 CPU 训练速度会明显较慢。

## 更新到 GitHub（不上传本地数据）

仓库通过 `.gitignore` 只同步代码、配置和文档，不上传本地生成的数据。以下内容会保留在当前电脑：

- `data/frames/raw/` 中的原始图片。
- `data/labeling/export/` 中的标注图片和 YOLO 标签。
- `data/dataset/images/` 和 `data/dataset/labels/` 中的训练数据集。
- `data/raw_videos/`、深度数据和采集清单。
- `runs/` 中的训练结果、推理结果和模型权重。
- `.pt`、`.onnx` 和 `.engine` 模型文件。

提交前可以运行：

```bash
cd /home/nina/stator-yolo
git status --short
```

输出中不应出现 JPG、PNG、标签 TXT、视频、数据集或模型文件。

WSL 首次向 GitHub 推送时，先登录：

```bash
gh auth login
```

依次选择：

```text
GitHub.com
HTTPS
Login with a web browser
```

之后更新代码：

```bash
git add .
git commit -m "说明本次更新内容"
git push origin main
```

如果显示 `nothing to commit`，表示没有新的本地代码改动；如果显示 `Everything up-to-date`，表示 GitHub 已是最新版本。

## 安装

进入 WSL 项目目录：

```bash
cd /home/nina/stator-yolo
```

创建并激活虚拟环境：

```bash
python3 -m venv .venv
source .venv/bin/activate
```

安装项目及依赖：

```bash
python -m pip install --upgrade pip
python -m pip install -e .
```

检查环境：

```bash
stator-yolo-env
```

检查 WSL 是否可以使用 NVIDIA GPU：

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'GPU 不可用')"
```

正常情况下应输出：

```text
True
NVIDIA GeForce RTX 5060 Ti
```

## 启动网站

激活虚拟环境后运行：

```bash
cd /home/nina/stator-yolo
source .venv/bin/activate
python run_web.py --host 0.0.0.0 --port 8000
```

也可以使用安装后的命令：

```bash
stator-yolo-web --host 0.0.0.0 --port 8000
```

看到下面的信息代表服务已经启动：

```text
Stator YOLO Web: http://0.0.0.0:8000
Press Ctrl+C to stop the server.
```

本机访问：

```text
http://127.0.0.1:8000
```

运行期间不要关闭 WSL 终端，也不要按 `Ctrl+C`。

## 局域网部署

局域网入口不是固定地址。Windows 的局域网 IP 可能在重启、重新连接 Wi-Fi
或路由器重新分配地址后发生变化，因此每次使用时应先查询当前地址，不要继续使用
README 或浏览器历史记录中的旧 IP。

网络链路：

```text
局域网浏览器
    ↓
Windows 当前局域网 IP:8001
    ↓
Windows portproxy
    ↓
WSL 127.0.0.1:8000
    ↓
Stator YOLO Web
```

首次部署时，以管理员身份打开 Windows PowerShell：

```powershell
cd "\\wsl$\Ubuntu-24.04\home\nina\stator-yolo"
powershell -ExecutionPolicy Bypass -File .\deploy_intranet.ps1
```

部署脚本会：

- 开放 Windows TCP 8001 入站规则。
- 限制访问来源为 `192.168.10.0/24`。
- 建立 `8001 → 8000` 端口转发。
- 注册登录后启动网站的 Windows 计划任务。

### 每次使用：检查并打开局域网网站

先在 WSL 终端确认网站服务已经启动：

```bash
cd /home/nina/stator-yolo
source .venv/bin/activate
python run_web.py --host 0.0.0.0 --port 8000
```

如果提示 `Address already in use`，通常表示服务已由计划任务启动，不需要重复启动。
可以用下面的命令确认本机网站是否正常；看到 `HTTP 200` 即正常：

```bash
curl -sS -o /dev/null -w "HTTP %{http_code}\n" http://127.0.0.1:8000/
```

然后打开 Windows PowerShell，运行以下命令。它会自动获取当前局域网 IP、检查
`8001` 端口，并在本机浏览器中打开正确的网址：

```powershell
$lanIp = Get-NetIPConfiguration |
    Where-Object { $_.IPv4DefaultGateway -and $_.NetAdapter.Status -eq "Up" } |
    ForEach-Object { $_.IPv4Address.IPAddress } |
    Select-Object -First 1
$url = "http://${lanIp}:8001"
Write-Host "局域网地址：$url"
Test-NetConnection 127.0.0.1 -Port 8001
Start-Process $url
```

将 PowerShell 显示的“局域网地址”发给其他设备。例如当前 IP 如果是
`192.168.10.13`，访问地址就是 `http://192.168.10.13:8001`。其他设备必须连接
同一个局域网；访客 Wi-Fi 或路由器的客户端隔离功能可能阻止设备互相访问。

如果 `TcpTestSucceeded` 显示 `False`，请以管理员身份打开 Windows PowerShell，
重新运行部署脚本：

```powershell
cd "\\wsl$\Ubuntu-24.04\home\nina\stator-yolo"
powershell -ExecutionPolicy Bypass -File .\deploy_intranet.ps1
```

### 安全说明

当前网站没有账号、密码和权限系统。能够访问网站的用户也可以：

- 上传或删除图片。
- 修改标签。
- 划分和增强数据集。
- 启动或停止训练。

因此只应在可信局域网中使用，不要直接映射到互联网。

## 标注流程

### 1. 上传图片

打开网站，点击左侧“导入图片”，选择一张或多张图片。

上传的源图片保存在：

```text
data/frames/raw/
```

### 2. 框选定子

在画布中按住鼠标拖拽，为图片中的每个定子绘制紧贴目标边界的矩形框。

当前只有一个类别：

```text
0: stator
```

标注规则：

- 框尽量贴合可见的定子边界。
- 仍能辨认的部分遮挡定子需要标注。
- 严重模糊、无法确认的目标不标注。
- 不要标注倒影或与定子相似的背景物体。

### 3. 保存标签

点击“保存标签”或“保存并下一张”。

保存后生成：

```text
data/labeling/export/
├── images/
└── labels/
```

标签采用标准 YOLO 检测格式：

```text
class_id x_center y_center width height
```

坐标全部归一化到 `[0, 1]`。

### 4. 删除图片

选择图片后点击“删除图片”。确认后会同时删除：

- `data/frames/raw/` 中的源图片。
- `data/labeling/export/images/` 中的导出图片。
- `data/labeling/export/labels/` 中对应的标签。

该操作目前不可撤销。

## 数据集处理

点击网站右上角“数据与训练”。

### 检查标签

“检查标签”会验证：

- 每张导出图片是否存在对应标签。
- 每行标签是否包含五个字段。
- 类别 ID 和坐标是否为数字。
- 坐标是否位于 `[0, 1]`。

对应命令行：

```bash
python scripts/check_yolo_labels.py \
  --images-dir data/labeling/export/images \
  --labels-dir data/labeling/export/labels
```

### 划分数据集

网页按以下比例划分：

- 训练集：80%
- 验证集：10%
- 测试集：10%

输出目录：

```text
data/dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

当前脚本是随机划分。正式评估时，建议按照采集场景或视频会话划分，避免相邻的近重复帧同时进入训练集和验证集。

### 数据增强

网页可以设置每张训练图片生成的增强副本数量。

增强只作用于训练集，可能包含：

- 亮度和对比度变化。
- 噪声。
- 运动模糊和高斯模糊。
- 小角度旋转。
- CLAHE 局部对比度增强。

验证集和测试集不会被增强。

## 模型训练

训练前必须满足：

- `data/dataset/images/train` 中存在图片。
- `data/dataset/labels/train` 中存在同名标签。
- 验证集中存在图片和标签。
- WSL 中 `torch.cuda.is_available()` 返回 `True`。

点击网站右上角“数据与训练”，配置：

- 训练轮数：首次测试建议 `10`，正式训练建议 `100`。
- 批量大小：RTX 5060 Ti 8 GB 建议从 `8` 开始。
- 图片尺寸：默认 `640`。

首次流程测试建议：

```text
epochs = 10
batch = 8
image size = 640
```

正式训练建议：

```text
epochs = 100
batch = 8
image size = 640
```

如果出现 CUDA 显存不足，将批量大小改为 `4` 或 `2`。

对应命令行：

```bash
EPOCHS=100 IMGSZ=640 BATCH=8 DEVICE=0 bash train/train_yolov8.sh
```

训练结果：

```text
runs/stator_yolov8/
├── weights/
│   ├── best.pt
│   └── last.pt
├── results.csv
└── results.png
```

其中 `best.pt` 是验证指标最好的模型，通常用于后续推理和导出。

## 推理

### 图片推理

```bash
python scripts/infer_image.py \
  --model runs/stator_yolov8/weights/best.pt \
  --image path/to/test.jpg \
  --output runs/infer/result.jpg \
  --conf 0.25
```

### 视频推理

```bash
python scripts/infer_video.py \
  --model runs/stator_yolov8/weights/best.pt \
  --video path/to/test.mp4 \
  --output runs/infer/result.mp4 \
  --conf 0.25
```

### RealSense RGB-D 推理

```bash
python scripts/infer_realsense_rgbd.py \
  --model runs/stator_yolov8/weights/best.pt
```

RealSense 推理会结合检测框中心附近的深度值，计算相机坐标系下的三维坐标。相机测试和采集细节见：

- `docs/workflow.md`
- `docs/capture_label_augment.md`

## TensorRT 导出

在带 NVIDIA GPU 和 TensorRT 环境的设备上运行：

```bash
bash export/export_engine.sh runs/stator_yolov8/weights/best.pt
```

导出的 `.engine` 与设备、CUDA、TensorRT 版本密切相关。建议在最终运行推理的 Jetson 或目标设备上导出，不要假设不同设备生成的 Engine 可以直接通用。

## ONNX 精简部署

远端机器不需要安装 PyTorch 和 Ultralytics 时，可以将最佳模型导出为 ONNX：

```bash
cd /home/nina/stator-yolo
source .venv/bin/activate

yolo export \
  model=runs/stator_yolov8/weights/best.pt \
  format=onnx \
  imgsz=960 \
  simplify=True
```

首次导出时，Ultralytics可能自动安装 `onnx`、`onnxruntime` 和 `onnxslim`。
看到 `ONNX: export success` 表示导出成功，模型位于：

```text
runs/stator_yolov8/weights/best.onnx
```

将模型复制到精简部署目录：

```bash
cp runs/stator_yolov8/weights/best.onnx deploy/onnx/best.onnx
```

然后将整个 `deploy/onnx/` 目录复制到远端机器。部署目录提供基于 ONNX
Runtime 的图片、视频和普通摄像头推理程序，具体命令见
`deploy/onnx/README.md`。

`best.onnx` 受 `.gitignore` 保护，不会上传到 GitHub，需要通过 U 盘、
`scp` 或内部文件服务器单独传输。

## 项目结构

```text
stator-yolo/
├── run_web.py                  # Web 服务启动入口
├── deploy_intranet.ps1         # Windows 局域网部署
├── pyproject.toml              # Python 包与依赖配置
├── stator_yolo/
│   ├── cli.py                  # 统一命令入口
│   ├── env_check.py            # 环境检查
│   ├── paths.py                # 项目目录管理
│   └── web.py                  # Web API 与任务控制
├── web/
│   ├── index.html              # Web 页面
│   ├── app.js                  # 标注和训练交互
│   └── styles.css              # 页面样式
├── scripts/
│   ├── check_yolo_labels.py    # 标签检查
│   ├── split_dataset.py        # 数据集划分
│   ├── augment_dataset.py      # 数据增强
│   ├── infer_image.py          # 图片推理
│   ├── infer_video.py          # 视频推理
│   └── infer_realsense_rgbd.py # RealSense RGB-D 推理
├── train/
│   └── train_yolov8.sh         # YOLO 训练
├── export/
│   └── export_engine.sh        # TensorRT 导出
├── deploy/
│   └── onnx/                   # 不依赖 PyTorch 的精简真机部署包
├── data/
│   ├── frames/raw/             # 上传或采集的源图片
│   ├── labeling/export/        # 已保存的图片和标签
│   ├── dataset/                # 训练/验证/测试数据集
│   └── dataset.yaml            # YOLO 数据集配置
└── runs/                       # 训练、推理和导出结果
```

## 常见问题

### 浏览器显示 `ERR_EMPTY_RESPONSE`

通常表示 Windows 的 `8001` 已打开，但 WSL 中的 Python 服务没有运行。

启动服务：

```bash
cd /home/nina/stator-yolo
source .venv/bin/activate
python run_web.py --host 0.0.0.0 --port 8000
```

检查 Windows 转发：

```powershell
netsh interface portproxy show v4tov4
```

应包含：

```text
0.0.0.0  8001  127.0.0.1  8000
```

### 端口被占用

查看占用进程：

```bash
sudo ss -ltnp | grep :8000
```

停止旧服务：

```bash
sudo fuser -k 8000/tcp
```

然后重新启动网站。

### 页面修改后没有变化

先重启 Python 服务，再在浏览器中按 `Ctrl+F5` 强制刷新。

### `favicon.ico` 返回 404

这只表示项目暂时没有浏览器页签图标，不影响网站功能。

### 训练提示没有图片

仅有标签不能训练。必须在网页中框选目标并点击“保存标签”，确保以下两个目录中存在同名文件：

```text
data/labeling/export/images/
data/labeling/export/labels/
```

随后重新执行“检查标签”和“划分数据集”。

### CUDA 显存不足

将批量大小从 `8` 调整为 `4` 或 `2`。必要时将图片尺寸从 `640` 降低为 `416`。

## 推荐工作顺序

```text
启动网站
  → 上传图片
  → 框选定子
  → 保存全部标签
  → 检查标签
  → 划分数据集
  → 增强训练集（可选）
  → 先训练 10 轮检查流程
  → 正式训练 100 轮
  → 使用 best.pt 推理
  → 在目标设备导出 TensorRT Engine
```
