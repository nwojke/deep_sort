# Deep SORT v2 - PyTorch版本

基于PyTorch的Deep SORT多目标跟踪系统升级版，支持多类别跟踪、端到端检测跟踪流程、EMA特征更新和轨迹插值。

## 新特性

### 1. PyTorch深度学习框架
- 使用PyTorch替代TensorFlow
- 实现了基于ResNet的特征提取器
- 支持GPU加速

### 2. 多类别跟踪 (Multi-Class Tracking)
- 支持同时跟踪多个目标类别
- 类别感知的轨迹关联
- 可配置的类别匹配权重

### 3. YOLOv8集成
- 集成Ultralytics YOLOv8检测器
- 支持视频文件 (.mp4) 和摄像头输入
- 端到端"目标检测 -> 深度特征提取 -> 轨迹关联"流程

### 4. EMA特征更新
- 使用指数移动平均(EMA)平滑目标特征
- 减少检测器误差引入的错误特征
- 提高特征表示的稳定性

### 5. 轨迹插值与平滑
- 离线轨迹插值填补遮挡期间的缺失帧
- 支持线性插值和样条插值
- 轨迹平滑减少抖动

## 安装

```bash
# 安装依赖
pip install -r requirements.txt

# 下载YOLOv8模型（首次使用会自动下载）
# 可选模型: yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt
```

## 快速开始

### 摄像头实时跟踪

```bash
python deep_sort_app_v2.py --source 0
```

### 视频文件跟踪

```bash
python deep_sort_app_v2.py \
    --source path/to/video.mp4 \
    --save-video \
    --output result.mp4
```

### 多类别跟踪（人和车辆）

```bash
python deep_sort_app_v2.py \
    --source path/to/video.mp4 \
    --classes 0 2 3 5 7 \
    --save-video
```

类别ID对应关系（COCO数据集）：
- 0: person (人)
- 2: car (汽车)
- 3: motorcycle (摩托车)
- 5: bus (公交车)
- 7: truck (卡车)

## 参数说明

### 输入输出
- `--source`: 视频文件路径或摄像头索引 (默认: 0)
- `--output`: 输出视频路径
- `--save-video`: 保存输出视频

### 模型设置
- `--model`: YOLOv8模型路径 (默认: yolov8n.pt)
- `--feature-model`: 特征提取器模型路径（可选）

### 检测参数
- `--conf-threshold`: 检测置信度阈值 (默认: 0.3)
- `--iou-threshold`: NMS IOU阈值 (默认: 0.45)
- `--classes`: 要跟踪的类别ID列表

### 跟踪参数
- `--max-cosine-distance`: 最大余弦距离 (默认: 0.2)
- `--nn-budget`: 外观描述符库大小 (默认: 100)
- `--max-age`: 轨迹最大存活帧数 (默认: 30)
- `--n-init`: 轨迹初始化所需检测次数 (默认: 3)
- `--ema-alpha`: EMA平滑因子 (默认: 0.9)
- `--mc-lambda`: 多类别匹配权重 (默认: 0.995)

### 后处理参数
- `--no-interpolation`: 禁用轨迹插值
- `--no-smoothing`: 禁用轨迹平滑
- `--max-interpolation-gap`: 最大插值间隔 (默认: 30)

## API使用示例

```python
from deep_sort import (
    Tracker,
    NearestNeighborDistanceMetric,
    YOLODetector,
    FeatureExtractor
)

# 创建检测器
detector = YOLODetector(
    model_path='yolov8n.pt',
    conf_threshold=0.3,
    classes=[0]  # 只检测人
)

# 创建特征提取器
feature_extractor = FeatureExtractor(device='cuda')

# 创建跟踪器
metric = NearestNeighborDistanceMetric("cosine", 0.2, 100)
tracker = Tracker(
    metric,
    max_iou_distance=0.7,
    max_age=30,
    n_init=3,
    ema_alpha=0.9,  # EMA平滑因子
    mc_lambda=0.995  # 多类别匹配权重
)

# 处理视频
import cv2
cap = cv2.VideoCapture('video.mp4')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 检测和跟踪
    results = detector.detect_and_track(frame, tracker, feature_extractor)

    # 处理结果
    for result in results:
        track_id = result['track_id']
        bbox = result['bbox']  # [x1, y1, x2, y2]
        class_id = result['class_id']
        class_name = result['class_name']

        print(f"Track {track_id}: {class_name} at {bbox}")

cap.release()
```

## 项目结构

```
deep_sort/
├── deep_sort/
│   ├── __init__.py
│   ├── detection.py          # 检测类
│   ├── track.py              # 轨迹类（含EMA）
│   ├── tracker.py            # 跟踪器（多类别支持）
│   ├── kalman_filter.py      # 卡尔曼滤波
│   ├── nn_matching.py        # 最近邻匹配
│   ├── linear_assignment.py  # 线性分配
│   ├── iou_matching.py       # IOU匹配
│   ├── deep_feature_extractor.py  # PyTorch特征提取器
│   ├── yolo_detector.py      # YOLOv8检测器
│   └── track_interpolation.py # 轨迹插值与平滑
├── deep_sort_app_v2.py       # 主应用
├── example_usage.py          # 使用示例
├── requirements.txt
└── README_V2.md
```

## 性能优化建议

1. **使用GPU**: 确保CUDA可用以加速特征提取
2. **选择合适的YOLO模型**: 
   - yolov8n.pt: 最快，精度较低
   - yolov8s.pt: 平衡
   - yolov8m/l/x.pt: 更慢，精度更高
3. **调整跟踪参数**:
   - 降低`max_age`可减少ID切换但可能丢失目标
   - 调整`ema_alpha`平衡特征更新速度

## 许可证

与原Deep SORT项目相同。
