#!/usr/bin/env python3
"""
Deep SORT v2 使用示例

演示如何使用新的PyTorch版本Deep SORT进行多目标跟踪。
"""

import cv2
from deep_sort import (
    Tracker,
    NearestNeighborDistanceMetric,
    YOLODetector,
    FeatureExtractor,
    TrackPostProcessor
)


def example_basic_tracking():
    """基本跟踪示例。"""
    print("=" * 50)
    print("示例1: 基本跟踪")
    print("=" * 50)

    # 创建检测器
    detector = YOLODetector(
        model_path='yolov8n.pt',
        conf_threshold=0.3,
        classes=[0]  # 只检测人
    )

    # 创建特征提取器
    feature_extractor = FeatureExtractor(device='cpu')

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

    # 打开视频
    video_path = 'path/to/your/video.mp4'  # 替换为你的视频路径
    cap = cv2.VideoCapture(video_path)

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        # 检测和跟踪
        results = detector.detect_and_track(frame, tracker, feature_extractor)

        # 打印结果
        print(f"Frame {frame_idx}: {len(results)} tracks")
        for r in results:
            print(f"  ID {r['track_id']}: {r['class_name']} at {r['bbox']}")

        # 可视化（可选）
        # ...

        if frame_idx >= 100:  # 只处理前100帧
            break

    cap.release()
    print("示例1完成\n")


def example_multi_class_tracking():
    """多类别跟踪示例。"""
    print("=" * 50)
    print("示例2: 多类别跟踪")
    print("=" * 50)

    # 创建检测器 - 检测多个类别
    detector = YOLODetector(
        model_path='yolov8n.pt',
        conf_threshold=0.3,
        classes=[0, 2, 3, 5, 7]  # 人、汽车、摩托车、公交车、卡车
    )

    feature_extractor = FeatureExtractor(device='cpu')

    metric = NearestNeighborDistanceMetric("cosine", 0.2, 100)
    tracker = Tracker(
        metric,
        max_iou_distance=0.7,
        max_age=30,
        n_init=3,
        ema_alpha=0.9,
        mc_lambda=0.995  # 启用多类别匹配
    )

    print("支持跟踪的类别:")
    for class_id, class_name in detector.class_names.items():
        if class_id in [0, 2, 3, 5, 7]:
            print(f"  {class_id}: {class_name}")

    print("\n示例2完成\n")


def example_with_interpolation():
    """带轨迹插值的跟踪示例。"""
    print("=" * 50)
    print("示例3: 带轨迹插值的跟踪")
    print("=" * 50)

    # 创建后处理器
    post_processor = TrackPostProcessor(
        enable_interpolation=True,
        enable_smoothing=True,
        max_gap=30,  # 最大插值间隔
        smooth_window=5  # 平滑窗口大小
    )

    # 模拟一些轨迹数据
    from collections import defaultdict
    track_history = defaultdict(list)

    # 模拟轨迹（实际使用时从跟踪器获取）
    for track_id in range(3):
        for frame_idx in range(100):
            if frame_idx % 10 != 5:  # 模拟丢失第5, 15, 25...帧
                track_history[track_id].append({
                    'frame_idx': frame_idx,
                    'bbox': [100 + track_id * 50 + frame_idx, 200, 50, 100],
                    'mean': [125 + track_id * 50 + frame_idx, 250, 0.5, 100, 0, 0, 0, 0],
                    'covariance': [[1, 0, 0, 0, 0, 0, 0, 0],
                                  [0, 1, 0, 0, 0, 0, 0, 0],
                                  [0, 0, 1, 0, 0, 0, 0, 0],
                                  [0, 0, 0, 1, 0, 0, 0, 0],
                                  [0, 0, 0, 0, 1, 0, 0, 0],
                                  [0, 0, 0, 0, 0, 1, 0, 0],
                                  [0, 0, 0, 0, 0, 0, 1, 0],
                                  [0, 0, 0, 0, 0, 0, 0, 1]],
                    'class_id': 0,
                    'class_name': 'person'
                })

    print(f"原始轨迹数: {len(track_history)}")
    for track_id, history in track_history.items():
        print(f"  Track {track_id}: {len(history)} frames")

    # 后处理
    processed_tracks = post_processor.process_frame_results(track_history)

    print(f"\n处理后轨迹数: {len(processed_tracks)}")
    for track_id, history in processed_tracks.items():
        interpolated_count = sum(1 for h in history if h.get('interpolated', False))
        print(f"  Track {track_id}: {len(history)} frames ({interpolated_count} interpolated)")

    print("\n示例3完成\n")


def example_camera_tracking():
    """摄像头实时跟踪示例。"""
    print("=" * 50)
    print("示例4: 摄像头实时跟踪")
    print("=" * 50)

    print("""
    要使用摄像头进行实时跟踪，请运行:

    python deep_sort_app_v2.py --source 0 --model yolov8n.pt

    参数说明:
    --source 0          : 使用默认摄像头
    --model yolov8n.pt  : YOLOv8模型
    --classes 0         : 只跟踪人
    --conf-threshold 0.3: 置信度阈值
    --max-age 30        : 轨迹最大存活帧数
    --ema-alpha 0.9     : EMA平滑因子
    --save-video        : 保存结果视频
    --output result.mp4 : 输出视频路径
    """)

    print("示例4完成\n")


def example_video_tracking():
    """视频文件跟踪示例。"""
    print("=" * 50)
    print("示例5: 视频文件跟踪")
    print("=" * 50)

    print("""
    要对视频文件进行跟踪，请运行:

    python deep_sort_app_v2.py \
        --source path/to/video.mp4 \
        --model yolov8n.pt \
        --classes 0 2 3 5 7 \
        --conf-threshold 0.3 \
        --iou-threshold 0.45 \
        --max-cosine-distance 0.2 \
        --nn-budget 100 \
        --max-age 30 \
        --n-init 3 \
        --ema-alpha 0.9 \
        --mc-lambda 0.995 \
        --save-video \
        --output result.mp4

    高级选项:
    --no-interpolation      : 禁用轨迹插值
    --no-smoothing          : 禁用轨迹平滑
    --max-interpolation-gap 30 : 最大插值间隔
    """)

    print("示例5完成\n")


def main():
    """运行所有示例。"""
    print("\n" + "=" * 50)
    print("Deep SORT v2 使用示例")
    print("=" * 50 + "\n")

    # 运行示例
    example_multi_class_tracking()
    example_with_interpolation()
    example_camera_tracking()
    example_video_tracking()

    print("=" * 50)
    print("所有示例完成!")
    print("=" * 50)


if __name__ == '__main__':
    main()
