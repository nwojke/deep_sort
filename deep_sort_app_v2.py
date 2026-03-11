#!/usr/bin/env python3
"""
Deep SORT v2 - 端到端多目标跟踪应用

支持：
1. PyTorch深度学习框架
2. 多类别跟踪
3. YOLOv8集成（视频流/摄像头输入）
4. EMA特征更新
5. 轨迹插值与平滑
"""

import argparse
import os
import time
from collections import defaultdict

import cv2
import numpy as np

from deep_sort import nn_matching
from deep_sort.tracker import Tracker
from deep_sort.yolo_detector import YOLODetector, VideoCapture, draw_tracks
from deep_sort.deep_feature_extractor import FeatureExtractor
from deep_sort.track_interpolation import TrackPostProcessor


def run_tracking(
    source,
    output_path=None,
    model='yolov8n.pt',
    feature_model=None,
    conf_threshold=0.3,
    iou_threshold=0.45,
    max_cosine_distance=0.2,
    nn_budget=100,
    max_age=30,
    n_init=3,
    ema_alpha=0.9,
    mc_lambda=0.995,
    display=True,
    save_video=False,
    classes=None,
    device=None,
    enable_interpolation=True,
    enable_smoothing=True,
    max_interpolation_gap=30
):
    """
    运行端到端跟踪。

    Parameters
    ----------
    source : str or int
        视频文件路径或摄像头索引。
    output_path : str, optional
        输出视频路径。
    model : str
        YOLOv8模型路径。
    feature_model : str, optional
        特征提取器模型路径。
    conf_threshold : float
        检测置信度阈值。
    iou_threshold : float
        NMS IOU阈值。
    max_cosine_distance : float
        最大余弦距离。
    nn_budget : int
        外观描述符库的最大大小。
    max_age : int
        轨迹最大存活帧数。
    n_init : int
        轨迹初始化所需检测次数。
    ema_alpha : float
        EMA平滑因子。
    mc_lambda : float
        多类别匹配权重。
    display : bool
        是否显示结果。
    save_video : bool
        是否保存视频。
    classes : list, optional
        要跟踪的类别列表。
    device : str, optional
        计算设备。
    enable_interpolation : bool
        是否启用轨迹插值。
    enable_smoothing : bool
        是否启用轨迹平滑。
    max_interpolation_gap : int
        最大插值间隔。
    """
    # 创建检测器
    print(f"Loading YOLO model: {model}")
    detector = YOLODetector(
        model_path=model,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
        device=device,
        classes=classes
    )

    # 创建特征提取器
    print("Initializing feature extractor...")
    feature_extractor = FeatureExtractor(
        model_path=feature_model,
        device=device if device else ('cuda' if cv2.cuda.getCudaEnabledDeviceCount() > 0 else 'cpu')
    )

    # 创建跟踪器
    metric = nn_matching.NearestNeighborDistanceMetric(
        "cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(
        metric,
        max_iou_distance=iou_threshold,
        max_age=max_age,
        n_init=n_init,
        ema_alpha=ema_alpha,
        mc_lambda=mc_lambda
    )

    # 创建轨迹后处理器
    post_processor = TrackPostProcessor(
        enable_interpolation=enable_interpolation,
        enable_smoothing=enable_smoothing,
        max_gap=max_interpolation_gap
    )

    # 打开视频源
    print(f"Opening video source: {source}")
    with VideoCapture(source) as cap:
        print(f"Video info: {cap.width}x{cap.height} @ {cap.fps:.2f}fps")

        # 设置视频写入器
        video_writer = None
        if save_video and output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(
                output_path, fourcc, cap.fps, (cap.width, cap.height))

        # 跟踪历史记录
        track_history = defaultdict(list)
        frame_idx = 0

        # 处理统计
        start_time = time.time()
        processing_times = []

        print("\nStarting tracking...")
        print("Press 'q' to quit, 'p' to pause\n")

        for frame in cap:
            frame_idx += 1
            loop_start = time.time()

            # 检测和跟踪
            results = detector.detect_and_track(
                frame, tracker, feature_extractor)

            # 记录轨迹历史
            for result in results:
                track_id = result['track_id']
                track_history[track_id].append({
                    'frame_idx': frame_idx,
                    'bbox': result['tlwh'],
                    'class_id': result['class_id'],
                    'class_name': result['class_name']
                })

            # 绘制结果
            display_frame = frame.copy()
            display_frame = draw_tracks(display_frame, results)

            # 添加统计信息
            elapsed = time.time() - loop_start
            processing_times.append(elapsed)
            avg_time = np.mean(processing_times[-30:])
            fps = 1.0 / avg_time if avg_time > 0 else 0

            info_text = [
                f"Frame: {frame_idx}",
                f"Tracks: {len(results)}",
                f"FPS: {fps:.1f}",
                f"Time: {elapsed*1000:.1f}ms"
            ]
            for i, text in enumerate(info_text):
                cv2.putText(display_frame, text, (10, 30 + i * 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # 保存视频
            if video_writer is not None:
                video_writer.write(display_frame)

            # 显示结果
            if display:
                cv2.imshow('Deep SORT v2', display_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('p'):
                    print("Paused. Press any key to continue...")
                    cv2.waitKey(0)

            # 打印进度
            if frame_idx % 30 == 0:
                print(f"Frame {frame_idx}: {len(results)} tracks, {fps:.1f} FPS")

        # 释放资源
        if video_writer is not None:
            video_writer.release()
        cv2.destroyAllWindows()

        # 后处理轨迹
        print("\nPost-processing tracks...")
        processed_tracks = post_processor.process_frame_results(track_history)

        # 输出统计
        total_time = time.time() - start_time
        print(f"\nTracking completed!")
        print(f"Total frames: {frame_idx}")
        print(f"Total time: {total_time:.2f}s")
        print(f"Average FPS: {frame_idx / total_time:.1f}")
        print(f"Unique tracks: {len(track_history)}")

        return processed_tracks


def run_on_video(
    video_path,
    output_path=None,
    **kwargs
):
    """在视频文件上运行跟踪。"""
    return run_tracking(
        source=video_path,
        output_path=output_path,
        **kwargs
    )


def run_on_camera(
    camera_id=0,
    output_path=None,
    **kwargs
):
    """在摄像头上运行跟踪。"""
    return run_tracking(
        source=camera_id,
        output_path=output_path,
        **kwargs
    )


def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description='Deep SORT v2 - End-to-End Tracking')

    # 输入输出
    parser.add_argument('--source', type=str, default='0',
                       help='视频文件路径或摄像头索引 (默认: 0)')
    parser.add_argument('--output', type=str, default=None,
                       help='输出视频路径')
    parser.add_argument('--save-video', action='store_true',
                       help='保存输出视频')

    # 模型设置
    parser.add_argument('--model', type=str, default='yolov8n.pt',
                       help='YOLOv8模型路径 (默认: yolov8n.pt)')
    parser.add_argument('--feature-model', type=str, default=None,
                       help='特征提取器模型路径')

    # 检测参数
    parser.add_argument('--conf-threshold', type=float, default=0.3,
                       help='检测置信度阈值 (默认: 0.3)')
    parser.add_argument('--iou-threshold', type=float, default=0.45,
                       help='NMS IOU阈值 (默认: 0.45)')
    parser.add_argument('--classes', type=int, nargs='+', default=None,
                       help='要跟踪的类别ID列表 (默认: 所有类别)')

    # 跟踪参数
    parser.add_argument('--max-cosine-distance', type=float, default=0.2,
                       help='最大余弦距离 (默认: 0.2)')
    parser.add_argument('--nn-budget', type=int, default=100,
                       help='外观描述符库大小 (默认: 100)')
    parser.add_argument('--max-age', type=int, default=30,
                       help='轨迹最大存活帧数 (默认: 30)')
    parser.add_argument('--n-init', type=int, default=3,
                       help='轨迹初始化所需检测次数 (默认: 3)')
    parser.add_argument('--ema-alpha', type=float, default=0.9,
                       help='EMA平滑因子 (默认: 0.9)')
    parser.add_argument('--mc-lambda', type=float, default=0.995,
                       help='多类别匹配权重 (默认: 0.995)')

    # 后处理参数
    parser.add_argument('--no-interpolation', action='store_true',
                       help='禁用轨迹插值')
    parser.add_argument('--no-smoothing', action='store_true',
                       help='禁用轨迹平滑')
    parser.add_argument('--max-interpolation-gap', type=int, default=30,
                       help='最大插值间隔 (默认: 30)')

    # 其他
    parser.add_argument('--no-display', action='store_true',
                       help='不显示结果')
    parser.add_argument('--device', type=str, default=None,
                       help='计算设备 (cuda/cpu)')

    return parser.parse_args()


def main():
    args = parse_args()

    # 解析输入源
    try:
        source = int(args.source)
        print(f"Using camera: {source}")
    except ValueError:
        source = args.source
        if not os.path.exists(source):
            raise ValueError(f"Video file not found: {source}")
        print(f"Using video file: {source}")

    # 运行跟踪
    run_tracking(
        source=source,
        output_path=args.output,
        model=args.model,
        feature_model=args.feature_model,
        conf_threshold=args.conf_threshold,
        iou_threshold=args.iou_threshold,
        max_cosine_distance=args.max_cosine_distance,
        nn_budget=args.nn_budget,
        max_age=args.max_age,
        n_init=args.n_init,
        ema_alpha=args.ema_alpha,
        mc_lambda=args.mc_lambda,
        display=not args.no_display,
        save_video=args.save_video,
        classes=args.classes,
        device=args.device,
        enable_interpolation=not args.no_interpolation,
        enable_smoothing=not args.no_smoothing,
        max_interpolation_gap=args.max_interpolation_gap
    )


if __name__ == '__main__':
    main()
