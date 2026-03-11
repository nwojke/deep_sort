import numpy as np
from . import kalman_filter
from . import linear_assignment
from . import iou_matching
from .track import Track


class Tracker:
    """
    多目标跟踪器，支持多类别跟踪。

    Parameters
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        测量到轨迹关联的距离度量。
    max_age : int
        轨迹被删除前的最大丢失次数。
    n_init : int
        轨迹被确认前的连续检测次数。如果在最初的 `n_init` 帧内发生丢失，
        轨迹状态被设置为 `Deleted`。
    max_iou_distance : float
        IOU关联的最大距离阈值。
    ema_alpha : float
        EMA特征更新的平滑因子。
    mc_lambda : float
        多类别跟踪的类别匹配权重。

    Attributes
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        用于测量到轨迹关联的距离度量。
    max_age : int
        轨迹被删除前的最大丢失次数。
    n_init : int
        轨迹保持在初始化阶段的帧数。
    kf : kalman_filter.KalmanFilter
        用于在图像空间中过滤目标轨迹的卡尔曼滤波器。
    tracks : List[Track]
        当前时间步的活跃轨迹列表。
    ema_alpha : float
        EMA特征更新的平滑因子。
    mc_lambda : float
        多类别跟踪的类别匹配权重。
    """

    def __init__(self, metric, max_iou_distance=0.7, max_age=30, n_init=3,
                 ema_alpha=0.9, mc_lambda=0.995):
        self.metric = metric
        self.max_iou_distance = max_iou_distance
        self.max_age = max_age
        self.n_init = n_init
        self.ema_alpha = ema_alpha
        self.mc_lambda = mc_lambda

        self.kf = kalman_filter.KalmanFilter()
        self.tracks = []
        self._next_id = 1

        # 用于轨迹插值的已删除轨迹缓存
        self._deleted_tracks_cache = []
        self.max_cache_size = 100

    def predict(self):
        """将轨迹状态分布向前传播一个时间步。

        此函数应在每个时间步调用一次，在 `update` 之前。
        """
        for track in self.tracks:
            track.predict(self.kf)

    def update(self, detections):
        """执行测量更新和轨迹管理。

        Parameters
        ----------
        detections : List[deep_sort.detection.Detection]
            当前时间步的检测列表。
        """
        # 运行匹配级联。
        matches, unmatched_tracks, unmatched_detections = \
            self._match(detections)

        # 更新轨迹集合。
        for track_idx, detection_idx in matches:
            self.tracks[track_idx].update(
                self.kf, detections[detection_idx])
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()

        # 缓存已删除的轨迹用于后续插值
        deleted_tracks = [t for t in self.tracks if t.is_deleted()]
        for track in deleted_tracks:
            if len(track.history) > 5:  # 只缓存有足够历史的轨迹
                self._deleted_tracks_cache.append(track)
        # 限制缓存大小
        if len(self._deleted_tracks_cache) > self.max_cache_size:
            self._deleted_tracks_cache = self._deleted_tracks_cache[-self.max_cache_size:]

        # 移除已删除的轨迹
        self.tracks = [t for t in self.tracks if not t.is_deleted()]

        for detection_idx in unmatched_detections:
            self._initiate_track(detections[detection_idx])

        # 更新距离度量。
        active_targets = [t.track_id for t in self.tracks if t.is_confirmed()]
        features, targets = [], []
        for track in self.tracks:
            if not track.is_confirmed():
                continue
            # 使用EMA平滑后的特征
            smoothed_feature = track.get_smoothed_feature()
            if smoothed_feature is not None:
                features.append(smoothed_feature)
                targets.append(track.track_id)
            track.features = []
        if len(features) > 0:
            self.metric.partial_fit(
                np.asarray(features), np.asarray(targets), active_targets)

    def _match(self, detections):
        """执行轨迹和检测之间的关联。"""

        def gated_metric(tracks, dets, track_indices, detection_indices):
            features = np.array([dets[i].feature for i in detection_indices])
            targets = np.array([tracks[i].track_id for i in track_indices])
            cost_matrix = self.metric.distance(features, targets)

            # 多类别：添加类别匹配成本
            for i, track_idx in enumerate(track_indices):
                track = tracks[track_idx]
                for j, det_idx in enumerate(detection_indices):
                    det = dets[det_idx]
                    if track.class_id != det.class_id:
                        # 类别不匹配时增加成本
                        cost_matrix[i, j] += (1 - self.mc_lambda) * 10

            cost_matrix = linear_assignment.gate_cost_matrix(
                self.kf, cost_matrix, tracks, dets, track_indices,
                detection_indices)

            return cost_matrix

        # 将轨迹集合分为已确认和未确认的轨迹。
        confirmed_tracks = [
            i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        unconfirmed_tracks = [
            i for i, t in enumerate(self.tracks) if not t.is_confirmed()]

        # 使用外观特征关联已确认的轨迹。
        matches_a, unmatched_tracks_a, unmatched_detections = \
            linear_assignment.matching_cascade(
                gated_metric, self.metric.matching_threshold, self.max_age,
                self.tracks, detections, confirmed_tracks)

        # 使用IOU关联剩余的轨迹和未确认的轨迹。
        iou_track_candidates = unconfirmed_tracks + [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update == 1]
        unmatched_tracks_a = [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update != 1]
        matches_b, unmatched_tracks_b, unmatched_detections = \
            linear_assignment.min_cost_matching(
                iou_matching.iou_cost, self.max_iou_distance, self.tracks,
                detections, iou_track_candidates, unmatched_detections)

        matches = matches_a + matches_b
        unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))
        return matches, unmatched_tracks, unmatched_detections

    def _initiate_track(self, detection):
        """从检测初始化新轨迹。"""
        mean, covariance = self.kf.initiate(detection.to_xyah())
        self.tracks.append(Track(
            mean, covariance, self._next_id, self.n_init, self.max_age,
            detection.feature, detection.class_id, detection.class_name,
            self.ema_alpha))
        self._next_id += 1

    def get_deleted_tracks_for_interpolation(self):
        """获取已删除的轨迹缓存，用于轨迹插值。

        Returns
        -------
        List[Track]
            已删除的轨迹列表。
        """
        return self._deleted_tracks_cache

    def clear_deleted_tracks_cache(self):
        """清空已删除轨迹缓存。"""
        self._deleted_tracks_cache = []
