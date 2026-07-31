# 定子 ONNX 真机部署包

该目录只包含远端推理所需内容，不包含标注、训练和数据集处理代码。

## 目录

```text
deploy/onnx/
├── best.onnx        # 本地生成，不提交 GitHub
├── infer.py         # 图片、视频和普通摄像头推理
├── requirements.txt
└── README.md
```

## 安装

```bash
cd deploy/onnx
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

默认安装 CPU 版 `onnxruntime`。如果远端具备兼容的 NVIDIA CUDA 环境，可将
它替换为 `onnxruntime-gpu`，并通过程序启动日志确认 Provider 为
`CUDAExecutionProvider`。

## 单张图片

```bash
python infer.py \
  --model best.onnx \
  --source test.jpg \
  --output result.jpg
```

## 视频

```bash
python infer.py \
  --model best.onnx \
  --source test.mp4 \
  --output result.mp4
```

## 普通摄像头

```bash
python infer.py --model best.onnx --source 0 --show
```

按 `Q` 或 `Esc` 退出实时预览。

默认置信度阈值为 `0.25`，可以使用 `--conf 0.5` 调整。该脚本使用
ONNX Runtime，不依赖 PyTorch 和 Ultralytics。

如果远端使用 RealSense 深度坐标、厂商相机 SDK 或机器人通信接口，需要在
`infer.py` 的图像输入与检测结果输出两端接入对应 SDK。
