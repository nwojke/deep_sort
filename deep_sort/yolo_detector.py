import numpy as np
import cv2
from ultralytics import YOLO


class YOLODetector:
    """
    YOLOv8检测器封装类，集成到Deep SORT中。

    Parameters
    ----------
    model_path : str
        YOLOv8模型路径或模型名称。
    conf_threshold : float
        置信度阈值。
    iou_threshold : float
        NMS的IOU阈值。
    device : str
        计算设备。
    classes : list, optional
        要检测的类别列表，None表示检测所有类别。

    Attributes
    ----------
    model : YOLO
        YOLOv8模型实例。
    conf_threshold : float
        置信度阈值。
    iou_threshold : float
        NMS的IOU阈值。
    classes : list
        要检测的类别列表。
    class_names : dict
        类别名称映射。
    """

    # COCO数据集的类别名称
    COCO_CLASSES = {
        0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane',
        5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light',
        10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench',
        14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow',
        20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack',
        25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee',
        30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite',
        34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard',
        37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass',
        41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl',
        46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli',
        51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake',
        56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table',
        61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote',
        66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven',
        70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock',
        75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier',
        79: 'toothbrush'
    }

    def __init__(self, model_path='yolov8n.pt', conf_threshold=0.3,
                 iou_threshold=0.45, device=None, classes=None):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.classes = classes
        self.device = device

        # 获取类别名称
        if hasattr(self.model, 'names'):
            self.class_names = self.model.names
        else:
            self.class_names = self.COCO_CLASSES

    def detect(self, image):
        """
        对图像进行目标检测。

        Parameters
        ----------
        image : ndarray
            输入图像，BGR格式。

        Returns
        -------
        tuple
            (boxes, confidences, class_ids, class_names)
            - boxes: 边界框数组，格式为 (x, y, w, h)
            - confidences: 置信度数组
            - class_ids: 类别ID数组
            - class_names: 类别名称列表
        """
        # 运行检测
        results = self.model(
            image,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            classes=self.classes,
            device=self.device,
            verbose=False
        )[0]

        boxes = []
        confidences = []
        class_ids = []
        class_names = []

        if results.boxes is not None:
            # 获取检测结果
            det_boxes = results.boxes.xyxy.cpu().numpy()  # xyxy格式
            det_confidences = results.boxes.conf.cpu().numpy()
            det_class_ids = results.boxes.cls.cpu().numpy().astype(int)

            for box, conf, cls_id in zip(det_boxes, det_confidences, det_class_ids):
                x1, y1, x2, y2 = box
                # 转换为xywh格式
                x = x1
                y = y1
                w = x2 - x1
                h = y2 - y1

                boxes.append([x, y, w, h])
                confidences.append(conf)
                class_ids.append(cls_id)
                class_names.append(self.class_names.get(cls_id, f"class_{cls_id}"))

        return np.array(boxes), np.array(confidences), np.array(class_ids), class_names

    def detect_and_track(self, image, tracker, feature_extractor=None):
        """
        检测目标并更新跟踪器。

        Parameters
        ----------
        image : ndarray
            输入图像。
        tracker : Tracker
            Deep SORT跟踪器实例。
        feature_extractor : FeatureExtractor, optional
            特征提取器。

        Returns
        -------
        list
            跟踪结果列表，每个元素为 (track_id, bbox, class_id, class_name)。
        """
        from .detection import Detection

        # 检测目标
        boxes, confidences, class_ids, class_names = self.detect(image)

        # 提取特征
        if feature_extractor is not None and len(boxes) > 0:
            features = feature_extractor(image, boxes)
        else:
            features = [None] * len(boxes)

        # 创建Detection对象
        detections = []
        for box, conf, cls_id, cls_name, feat in zip(
            boxes, confidences, class_ids, class_names, features):
            detections.append(Detection(
                tlwh=box,
                confidence=conf,
                feature=feat,
                class_id=cls_id,
                class_name=cls_name
            ))

        # 更新跟踪器
        tracker.predict()
        tracker.update(detections)

        # 收集跟踪结果
        results = []
        for track in tracker.tracks:
            if track.is_confirmed() and track.time_since_update == 0:
                bbox = track.to_tlbr()  # 使用tlbr格式用于可视化
                results.append({
                    'track_id': track.track_id,
                    'bbox': bbox,
                    'class_id': track.class_id,
                    'class_name': track.class_name,
                    'tlwh': track.to_tlwh()
                })

        return results


class VideoCapture:
    """
    视频捕获类，支持视频文件和摄像头输入。
    """

    def __init__(self, source):
        """
        Parameters
        ----------
        source : str or int
            视频文件路径或摄像头索引。
        """
        self.source = source
        self.cap = None
        self.frame_count = 0
        self.fps = 0
        self.width = 0
        self.height = 0

    def open(self):
        """打开视频源。"""
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            raise ValueError(f"无法打开视频源: {self.source}")

        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        return self

    def read(self):
        """读取一帧。"""
        if self.cap is None:
            return False, None
        return self.cap.read()

    def release(self):
        """释放视频源。"""
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def __enter__(self):
        return self.open()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
        return False

    def __iter__(self):
        """支持迭代。"""
        return self

    def __next__(self):
        """获取下一帧。"""
        ret, frame = self.read()
        if not ret:
            raise StopIteration
        return frame


def draw_tracks(image, tracks, show_class=True, show_id=True):
    """
    在图像上绘制跟踪结果。

    Parameters
    ----------
    image : ndarray
        输入图像。
    tracks : list
        跟踪结果列表。
    show_class : bool
        是否显示类别名称。
    show_id : bool
        是否显示跟踪ID。

    Returns
    -------
    ndarray
        绘制后的图像。
    """
    # 为不同类别生成颜色
    np.random.seed(42)
    colors = np.random.randint(0, 255, size=(100, 3), dtype=np.uint8)

    for track in tracks:
        track_id = track['track_id']
        bbox = track['bbox']
        class_id = track['class_id']
        class_name = track['class_name']

        x1, y1, x2, y2 = bbox.astype(int)

        # 使用类别ID选择颜色
        color = tuple(int(c) for c in colors[class_id % len(colors)])

        # 绘制边界框
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        # 绘制标签
        label_parts = []
        if show_id:
            label_parts.append(f"ID:{track_id}")
        if show_class:
            label_parts.append(class_name)

        if label_parts:
            label = " ".join(label_parts)
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            label_y = max(y1, label_size[1] + 10)

            cv2.rectangle(image, (x1, y1 - label_size[1] - 10),
                         (x1 + label_size[0], y1), color, -1)
            cv2.putText(image, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return image
