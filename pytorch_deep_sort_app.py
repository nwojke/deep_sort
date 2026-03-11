import cv2
import numpy as np
import argparse
from deep_sort.tracker import Tracker
from deep_sort.nn_matching import NearestNeighborDistanceMetric
from deep_sort.detection import Detection
from deep_sort.track_interpolator import TrackInterpolator
from tools.pytorch_feature_extractor import create_pytorch_box_encoder
from ultralytics import YOLO

class DeepSORTApp:
    def __init__(self, model_name='yolov8n', feature_dim=128, max_cosine_distance=0.2, max_gap=30):
        # 加载YOLOv8模型
        self.model = YOLO(model_name + '.pt')
        
        # 创建特征提取器
        self.encoder = create_pytorch_box_encoder(feature_dim=feature_dim)
        
        # 创建距离度量
        self.metric = NearestNeighborDistanceMetric('cosine', max_cosine_distance, feature_dim)
        
        # 创建跟踪器
        self.tracker = Tracker(self.metric)
        
        # 创建轨迹插值器
        self.interpolator = TrackInterpolator(max_gap=max_gap)
        
        # 类别名称映射
        self.class_names = self.model.names
        
        # 帧计数器
        self.frame_id = 0
    
    def process_frame(self, frame):
        # 使用YOLOv8进行目标检测
        results = self.model(frame, verbose=False)
        
        # 提取检测结果
        detections = []
        boxes = []
        confidences = []
        class_ids = []
        
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = box.conf[0].cpu().numpy()
                class_id = int(box.cls[0].cpu().numpy())
                
                # 转换为tlwh格式
                tlwh = [x1, y1, x2 - x1, y2 - y1]
                boxes.append(tlwh)
                confidences.append(confidence)
                class_ids.append(class_id)
        
        # 提取特征
        if boxes:
            features = self.encoder(frame, boxes)
        else:
            features = []
        
        # 创建Detection对象
        detections = []
        for i in range(len(boxes)):
            detection = Detection(boxes[i], confidences[i], features[i], class_ids[i])
            detections.append(detection)
        
        # 预测跟踪器状态
        self.tracker.predict()
        
        # 更新跟踪器
        self.tracker.update(detections)
        
        # 记录轨迹信息到插值器
        for track in self.tracker.tracks:
            if track.is_confirmed():
                tlwh = track.to_tlwh()
                self.interpolator.add_track(track.track_id, self.frame_id, tlwh, track.class_id)
        
        # 绘制当前跟踪结果（不使用插值）
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            
            # 获取边界框
            tlwh = track.to_tlwh()
            x1, y1, w, h = tlwh
            x2, y2 = x1 + w, y1 + h
            
            # 获取类别名称
            class_name = self.class_names.get(track.class_id, 'Unknown')
            
            # 绘制边界框和ID
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f'{class_name} {track.track_id}', 
                        (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
        
        # 增加帧计数器
        self.frame_id += 1
        
        return frame
    
    def run(self, input_source):
        # 打开视频流或摄像头
        if input_source.isdigit():
            cap = cv2.VideoCapture(int(input_source))
        else:
            cap = cv2.VideoCapture(input_source)
        
        if not cap.isOpened():
            print(f"Error: Could not open input source {input_source}")
            return
        
        # 处理视频流
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 处理帧
            frame = self.process_frame(frame)
            
            # 显示结果
            cv2.imshow('DeepSORT Tracking', frame)
            
            # 按'q'退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # 执行离线轨迹插值
        print("Performing offline track interpolation...")
        interpolated_history = self.interpolator.offline_interpolate_tracks()
        
        # 保存插值结果
        output_file = "interpolated_tracks.npy"
        self.interpolator.save_interpolated_tracks(output_file)
        print(f"Interpolated tracks saved to {output_file}")
        
        # 释放资源
        cap.release()
        cv2.destroyAllWindows()
        
        # 打印插值统计信息
        total_tracks = len(interpolated_history)
        total_points = sum(len(points) for points in interpolated_history.values())
        original_points = sum(len(points) for points in self.interpolator.track_history.values())
        interpolated_points = total_points - original_points
        print(f"Offline interpolation completed:")
        print(f"Total tracks: {total_tracks}")
        print(f"Original points: {original_points}")
        print(f"Interpolated points: {interpolated_points}")
        print(f"Total points after interpolation: {total_points}")

def parse_args():
    parser = argparse.ArgumentParser(description="DeepSORT with YOLOv8")
    parser.add_argument('--input', default='0', help='Input source (camera index or video file)')
    parser.add_argument('--model', default='yolov8n', help='YOLOv8 model name')
    parser.add_argument('--feature-dim', type=int, default=128, help='Feature dimension')
    parser.add_argument('--max-cosine-distance', type=float, default=0.2, help='Maximum cosine distance for matching')
    parser.add_argument('--max-gap', type=int, default=30, help='Maximum gap for track interpolation')
    return parser.parse_args()

def main():
    args = parse_args()
    app = DeepSORTApp(
        model_name=args.model,
        feature_dim=args.feature_dim,
        max_cosine_distance=args.max_cosine_distance,
        max_gap=args.max_gap
    )
    app.run(args.input)

if __name__ == "__main__":
    main()