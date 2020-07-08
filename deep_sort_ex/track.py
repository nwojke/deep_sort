# vim: expandtab:ts=4:sw=4

import queue
import numpy as np
from scipy.signal import butter, lfilter, lfilter_zi, freqz
# 滤波方案0
class Filter0(object):    
    '''均值滤波，设定队列长度，取队列均值作为输出
    '''
    def __init__(self, N=3):
        '''
        @param N - 队列长度
        '''
        self.q = queue.Queue()
        self.N = N

    def filter(self, data):
        '''
        @param data - np.array: channels x data_len
        '''
        for i in range(data.shape[1]):
            self.q.put(data[:, i])
            data[:, i] = np.array(self.q.queue).mean(axis=0)
            if self.q.qsize()>=self.N:
                self.q.get()

# 滤波方案1
class Filter1(object):
    '''依据方差剔除奇点数据
    '''
    def __init__(self, N=4, std_th=0.05, percent=0.8):
        '''
        @param N       - 队列长度
        @param std_th  - 方差域值
        @param percent - 奇异点保留前置能量比，当设为1.0即为完全用前置点替换奇异点
        '''
        self.q_size = N
        self.std_th = std_th
        self.percent = percent
        self.q = queue.Queue()

    def filter(self, data):
        '''
        @param data - np.array: channels x data_len
        '''
        for i in range(data.shape[1]):
            if self.q.qsize()==self.q_size:
                q_data = np.array(self.q.queue)
                mean_vals = np.array(q_data).mean(axis=0)
                err_vals = [data[:, i]-mean_vals]/mean_vals
                filter_ids = err_vals>self.std_th
                data[filter_ids[0], i] = mean_vals[filter_ids[0]]*self.percent + data[filter_ids[0], i]*(1.0-self.percent)
            self.q.put(data[:, i])
            if self.q.qsize()>self.q_size:
                self.q.get()

# 滤波方案2
class Filter2(object):
    '''简易的卡尔曼滤波
    '''
    def __init__(self, Q=1e-6, R=4e-4):
        '''
        @param Q - Q参数, channel x 1
        @param R - R参数, channel x 1
        '''
        self.Q = Q
        self.R = R
        self.K_prev = np.zeros_like(Q)
        self.X_prev = np.zeros_like(Q)
        self.P_prev = np.zeros_like(Q)
        self.b_first = True

    def filter(self, data):
        '''
        @param data - [InPlace], channel x data_len
        '''
        if self.b_first:
            self.b_first = False
            self.X_prev = data[:,0]
            self.P_prev = np.zeros_like(self.X_prev)
        else:
            self.K_prev = self.P_prev / (self.P_prev + self.R)
            data[:, 0] = self.X_prev + self.K_prev * (data[:, 0] - self.X_prev)
            self.P_prev = self.P_prev - self.K_prev * self.P_prev + self.Q
        for i in range(data.shape[1]):
            K = self.P_prev / (self.P_prev + self.R)
            data[:, i] = data[:, i-1] + K * (data[:, i] - data[:, i-1])
            P = self.P_prev - K * self.P_prev + self.Q
            self.P_prev = P
            self.K_prev = K
            self.X_prev = data[:, i]

# 滤波方案3_del
class Filter3_del(object):
    '''集成奇异值过滤和卡尔曼滤波
    '''
    def __init__(self, N=4, std_th=0.05, percent=0.8, Q=1e-6, R=4e-4, fs=5., cutoff=1.0, order=5):
        '''
        @param N       - 队列长度
        @param std_th  - 方差域值
        @param percent - 奇异点保留前置能量比，当设为1.0即为完全用前置点替换奇异点
        @param Q - Q参数, channel x 1
        @param R - R参数, channel x 1
        @param fs       - 采样率
        @param cutoff   - 截止频率, Hz
        @param order    - 滤波器阶数

        '''
        self.filter1 = Filter1(N=N, std_th=std_th, percent=percent)
        self.filter2 = Filter2(Q=Q, R=R)
        self.filter4 = Filter4(fs=fs, cutoff=cutoff, order=order)

    def filter(self, data):
        '''
        @param data - np.array: channels x data_len
        '''
        self.filter1.filter(data)
        self.filter2.filter(data)
        self.filter4.filter(data)

# 滤波方案3
class Filter3(object):
    '''Buffer低通滤波器
    '''
    def __init__(self, fs, cutoff=2.0, order=5):
        '''
        @param fs       - 采样率
        @param cutoff   - 截止频率, Hz
        @param order    - 滤波器阶数
        '''
        self.order = order
        b, a = self.butter_lowpass(cutoff, fs, order)
        self.b = b 
        self.a = a
        self.zi = lfilter_zi(b, a)
        self.index = 0

    def butter_lowpass(self, cutoff, fs, order=5):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return b, a

    def filter(self, data):
        '''
        @param data - [InPlace], channel x data_len
        '''
        index = self.index
        for data_chl in data: 
            for i in range(len(data_chl)):
                z, self.zi = lfilter(self.b, self.a, [data_chl[i]], zi=self.zi)
                if index+i<self.order:
                    continue
                data_chl[i] = z
        self.index += data.shape[0]
    

# 滤波方案总成
class Filter(object):
    def __init__(self, filter_type=4, q_size=4, std_th=0.05, percent=0.8, Q=1e-6, R=4e-4, fs=5, cutoff=2.0, order=5):
        '''
        @param fitler_type - 滤波器方案,对应 FilterX
        '''
        self.filters = []
        if filter_type&(1<<0):
            self.filters.append(Filter0(N=q_size))
        if filter_type&(1<<1):
            self.filters.append(Filter1(N=q_size, std_th=std_th, percent=percent))
        if filter_type&(1<<2):
            self.filters.append(Filter2(Q=Q, R=R))
        if filter_type&(1<<3):
            self.filters.append(Filter3(fs=fs, cutoff=cutoff, order=order))

    def filter(self, det):
        '''
        @param det - det.exts2: [d1, d2,....]
        '''
        if not det.exts2 is None:
            data = np.expand_dims(det.exts2, axis=1)
            for f in self.filters:
                f.filter(data)
            det.exts2 = data.reshape(-1)

    def filter_data(self, data):
        '''
        @param data - data: dim2 array
        '''
        for f in self.filters:
            f.filter(data)
        return data


class TrackState:
    """
    Enumeration type for the single target track state. Newly created tracks are
    classified as `tentative` until enough evidence has been collected. Then,
    the track state is changed to `confirmed`. Tracks that are no longer alive
    are classified as `deleted` to mark them for removal from the set of active
    tracks.

    """

    Tentative = 1
    Confirmed = 2
    Deleted = 3


class Track:
    """
    A single target track with state space `(x, y, a, h)` and associated
    velocities, where `(x, y)` is the center of the bounding box, `a` is the
    aspect ratio and `h` is the height.

    Parameters
    ----------
    mean : ndarray
        Mean vector of the initial state distribution.
    covariance : ndarray
        Covariance matrix of the initial state distribution.
    track_id : int
        A unique track identifier.
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.
    max_age : int
        The maximum number of consecutive misses before the track state is
        set to `Deleted`.
    feature : Optional[ndarray]
        Feature vector of the detection this track originates from. If not None,
        this feature is added to the `features` cache.

    Attributes
    ----------
    mean : ndarray
        Mean vector of the initial state distribution.
    covariance : ndarray
        Covariance matrix of the initial state distribution.
    track_id : int
        A unique track identifier.
    hits : int
        Total number of measurement updates.
    age : int
        Total number of frames since first occurance.
    time_since_update : int
        Total number of frames since last measurement update.
    state : TrackState
        The current track state.
    features : List[ndarray]
        A cache of features. On each measurement update, the associated feature
        vector is added to this list.

    """

    def __init__(self, mean, covariance, track_id, n_init, max_age,
                 feature=None, binding_obj=None, filter_type=0, q_size=4, std_th=0.05, percent=0.8, Q=1e-6, R=4e-4, fs=5., cutoff=1., order=5, save_to=None):
        '''
        扩展属性
        -----
        @param n_extend    - mean扩展属性数目
        @param filter_type - [Track] exts2滤波器类型
        @param q_size      - [Track] 队列长度
        @param std_th      - [Track] 相对误差域值
        @param percent     - [Track] 奇异点保留前置能量比，当设为1.0即为完全用前置点替换奇异点
        @param Q           - [Track] 卡尔曼滤波器参数
        @param R           - [Track] 卡尔曼滤波器参数
        @param fs          - [Track] 采样率
        @param cutoff      - [Track] 截止频率, Hz
        @param order       - [Track] 滤波器阶数
        @param save_to     - [Track] 采集数据保存目录
        '''

        self.mean = mean
        self.covariance = covariance
        self.track_id = track_id
        self.hits = 1
        self.age = 1
        self.time_since_update = 0

        self.state = TrackState.Tentative
        self.features = []
        if feature is not None:
            self.features.append(feature)

        self._n_init = n_init
        self._max_age = max_age
        self.binding_obj = binding_obj
        self.objFilter = Filter(filter_type=filter_type, q_size=q_size, std_th=std_th, Q=Q, R=R, fs=fs, cutoff=cutoff, order=order)
        self.exts2 = None
        self.save_to = save_to
        if not save_to is None:
            import os
            os.makedirs(save_to, exist_ok=True)

    def to_tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
        width, height)`.

        Returns
        -------
        ndarray
            The bounding box.

        """
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:4] / 2
        return ret

    def to_tlbr(self):
        """Get current position in bounding box format `(min x, miny, max x,
        max y)`.

        Returns
        -------
        ndarray
            The bounding box.

        """
        ret = self.to_tlwh()
        ret[2:4] = ret[:2] + ret[2:4]
        return ret

    def get_exts1(self):
        return self.mean[4:].copy()

    def get_exts2(self):
        return np.array(self.exts2)

    def predict(self, kf):
        """Propagate the state distribution to the current time step using a
        Kalman filter prediction step.

        Parameters
        ----------
        kf : kalman_filter.KalmanFilter
            The Kalman filter.

        """
        self.mean, self.covariance = kf.predict(self.mean, self.covariance)
        self.age += 1
        self.time_since_update += 1

    def update(self, kf, detection):
        """Perform Kalman filter measurement update step and update the feature
        cache.

        Parameters
        ----------
        kf : kalman_filter.KalmanFilter
            The Kalman filter.
        detection : Detection
            The associated detection.

        """
        # 数据采集: 滤波前
        # ---------------
        if not self.save_to is None:
            with open('%s/mean-orign-%s-%d.txt' % (self.save_to, detection.flag, self.track_id), 'a+') as f:
                f.write(str(list(detection.to_xyah()))[1:-1])
                f.write('\n')
            with open('%s/exts2-orign-%s-%d.txt' % (self.save_to, detection.flag, self.track_id), 'a+') as f:
                f.write(str(list(detection.exts2))[1:-1])
                f.write('\n')

        # 跟踪/滤波处理
        # ------------
        self.mean, self.covariance = kf.update(
            self.mean, self.covariance, detection.to_xyah())
        self.features.append(detection.feature)
        self.objFilter.filter(detection)
        self.exts2 = detection.exts2

        # 数据采集: 滤波后
        # ---------------
        if not self.save_to is None:
            with open('%s/mean-filter-%s-%d.txt' % (self.save_to, detection.flag, self.track_id), 'a+') as f:
                f.write(str(list(self.mean))[1:-1])
                f.write('\n')
            with open('%s/exts2-filter-%s-%d.txt' % (self.save_to, detection.flag, self.track_id), 'a+') as f:
                f.write(str(list(self.exts2))[1:-1])
                f.write('\n')

        # 状态更新
        # --------
        self.hits += 1
        self.time_since_update = 0
        if self.state == TrackState.Tentative and self.hits >= self._n_init:
            self.state = TrackState.Confirmed

    def mark_missed(self):
        """Mark this track as missed (no association at the current time step).
        """
        if self.state == TrackState.Tentative:
            self.state = TrackState.Deleted
        elif self.time_since_update > self._max_age:
            self.state = TrackState.Deleted

    def is_tentative(self):
        """Returns True if this track is tentative (unconfirmed).
        """
        return self.state == TrackState.Tentative

    def is_confirmed(self):
        """Returns True if this track is confirmed."""
        return self.state == TrackState.Confirmed

    def is_deleted(self):
        """Returns True if this track is dead and should be deleted."""
        return self.state == TrackState.Deleted
