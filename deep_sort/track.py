import numpy as np


class TrackState:
    """
    单目标跟踪状态的枚举类型。新创建的轨迹被分类为 `tentative`
    直到收集到足够的证据。然后，轨迹状态变为 `confirmed`。
    不再活跃的轨迹被分类为 `deleted`，标记为从活跃轨迹集合中移除。
    """
    Tentative = 1
    Confirmed = 2
    Deleted = 3


class Track:
    """
    单目标跟踪，状态空间为 `(x, y, a, h)` 和相关速度，
    其中 `(x, y)` 是边界框中心位置，`a` 是宽高比，`h` 是高度。

    Parameters
    ----------
    mean : ndarray
        初始状态分布的均值向量。
    covariance : ndarray
        初始状态分布的协方差矩阵。
    track_id : int
        唯一的轨迹标识符。
    n_init : int
        轨迹被确认前的连续检测次数。如果在最初的 `n_init` 帧内发生丢失，
        轨迹状态被设置为 `Deleted`。
    max_age : int
        轨迹状态被设置为 `Deleted` 前的最大连续丢失次数。
    feature : Optional[ndarray]
        此轨迹来源检测的特征向量。如果不为 None，此特征被添加到 `features` 缓存。
    class_id : int
        目标类别ID。
    class_name : str
        目标类别名称。
    ema_alpha : float
        EMA (指数移动平均) 的平滑因子。

    Attributes
    ----------
    mean : ndarray
        状态分布的均值向量。
    covariance : ndarray
        状态分布的协方差矩阵。
    track_id : int
        唯一的轨迹标识符。
    hits : int
        测量更新的总次数。
    age : int
        自首次出现以来的总帧数。
    time_since_update : int
        自上次测量更新以来的总帧数。
    state : TrackState
        当前轨迹状态。
    features : List[ndarray]
        特征缓存。每次测量更新时，关联的特征向量被添加到此列表。
    ema_feature : ndarray
        使用EMA更新的平滑特征表示。
    class_id : int
        目标类别ID。
    class_name : str
        目标类别名称。
    """

    def __init__(self, mean, covariance, track_id, n_init, max_age,
                 feature=None, class_id=0, class_name=None, ema_alpha=0.9):
        self.mean = mean
        self.covariance = covariance
        self.track_id = track_id
        self.hits = 1
        self.age = 1
        self.time_since_update = 0

        self.state = TrackState.Tentative
        self.features = []
        self.ema_alpha = ema_alpha

        # 使用EMA初始化特征
        if feature is not None:
            self.features.append(feature)
            self.ema_feature = feature.copy()
        else:
            self.ema_feature = None

        self._n_init = n_init
        self._max_age = max_age
        self.class_id = class_id
        self.class_name = class_name

        # 用于轨迹插值的历史记录
        self.history = []
        self._record_state()

    def _record_state(self):
        """记录当前状态到历史记录，用于轨迹插值。"""
        self.history.append({
            'mean': self.mean.copy(),
            'covariance': self.covariance.copy(),
            'age': self.age,
            'time_since_update': self.time_since_update,
            'state': self.state
        })

    def to_tlwh(self):
        """获取当前位置，格式为边界框 `(top left x, top left y, width, height)`.

        Returns
        -------
        ndarray
            边界框。
        """
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    def to_tlbr(self):
        """获取当前位置，格式为边界框 `(min x, min y, max x, max y)`.

        Returns
        -------
        ndarray
            边界框。
        """
        ret = self.to_tlwh()
        ret[2:] = ret[:2] + ret[2:]
        return ret

    def predict(self, kf):
        """使用卡尔曼滤波预测步骤将状态分布传播到当前时间步。

        Parameters
        ----------
        kf : kalman_filter.KalmanFilter
            卡尔曼滤波器。
        """
        self.mean, self.covariance = kf.predict(self.mean, self.covariance)
        self.age += 1
        self.time_since_update += 1
        self._record_state()

    def update(self, kf, detection):
        """执行卡尔曼滤波测量更新步骤并更新特征缓存。

        Parameters
        ----------
        kf : kalman_filter.KalmanFilter
            卡尔曼滤波器。
        detection : Detection
            关联的检测。
        """
        self.mean, self.covariance = kf.update(
            self.mean, self.covariance, detection.to_xyah())

        # 使用EMA更新特征
        if detection.feature is not None:
            self.features.append(detection.feature)
            if self.ema_feature is None:
                self.ema_feature = detection.feature.copy()
            else:
                self.ema_feature = (
                    self.ema_alpha * self.ema_feature +
                    (1 - self.ema_alpha) * detection.feature
                )

        self.hits += 1
        self.time_since_update = 0
        if self.state == TrackState.Tentative and self.hits >= self._n_init:
            self.state = TrackState.Confirmed

        self._record_state()

    def mark_missed(self):
        """将此轨迹标记为丢失（当前时间步无关联）。"""
        if self.state == TrackState.Tentative:
            self.state = TrackState.Deleted
        elif self.time_since_update > self._max_age:
            self.state = TrackState.Deleted

    def is_tentative(self):
        """如果此轨迹是 tentative（未确认），返回 True。"""
        return self.state == TrackState.Tentative

    def is_confirmed(self):
        """如果此轨迹已确认，返回 True。"""
        return self.state == TrackState.Confirmed

    def is_deleted(self):
        """如果此轨迹已死亡并应被删除，返回 True。"""
        return self.state == TrackState.Deleted

    def get_smoothed_feature(self):
        """获取EMA平滑后的特征表示。

        Returns
        -------
        ndarray
            EMA平滑后的特征向量。
        """
        return self.ema_feature if self.ema_feature is not None else (
            self.features[-1] if self.features else None
        )
