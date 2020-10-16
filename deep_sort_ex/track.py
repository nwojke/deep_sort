# vim: expandtab:ts=4:sw=4

import os
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
            if np.isnan(data[:, i]).any() or (data[:, i]==9999.0).any():
                if self.q.qsize()>0:
                    data[:, i] = np.array(self.q.queue).mean(axis=0)
                    self.q.put(data[:, i])
            else:
                self.q.put(data[:, i])
                data[:, i] = np.array(self.q.queue).mean(axis=0)
            if self.q.qsize()>=self.N:
                self.q.get()

# 滤波方案1
class Filter1(object):
    '''依据方差剔除奇点数据
        根据方差统计动态调整percent值
    设置队列记录过去N个数据值 queue[N]
    计算标准差 std(queue) => std_val
    动态计算percnet: 标准差换算
        相对误差 err = (det-mean)/mean
        设定sigmod曲线表 tbl_percent = np.exp(range(-100, 1))
        设定相对误差最大阈值 std_val = 0.1
        相对误差换算曲线表序号 tbl_index = int(err*(100/std_val))
        tbl_index修正处理: >100 => 设置为100
        percent = tbl_percent[tbl_index]
    
    '''
    def __init__(self, N=4, std_th=0.1, percent=0.8):
        '''
        @param N       - 队列长度
        @param std_th  - 方差阈值
        @param percent - [弃用，用sigmod自适应替代]奇异点保留前置能量比，当设为1.0即为完全用前置点替换奇异点
        '''
        self.q_size = N
        self.std_th = std_th
        self.percent = percent
        self.q = queue.Queue()
        self.tbl_percent = np.exp(range(-100, 1)) # 
        self.max_val = 100/std_th

    def filter(self, data):
        '''
        @param data - np.array: channels x data_len
        '''
        for i in range(data.shape[1]):
            exist_nan = False
            if np.isnan(data[:, i]).any() or (data[:, i]==9999.0).any():
                if self.q.qsize()>0:
                    data[:, i] = np.array(self.q.queue).mean(axis=0)
                exist_nan = True
            if self.q.qsize()==self.q_size:
                q_data = np.array(self.q.queue)
                mean_vals = np.array(q_data).mean(axis=0)
                err_vals = abs(data[:, i]-mean_vals)/abs(mean_vals)
                percent_ids = (err_vals*self.max_val).astype(np.int) # 相对误差截止点 0.1
                percent_ids[percent_ids>100]=100             # sigmod曲线
                percents = self.tbl_percent[percent_ids]
                data[:, i] = mean_vals*percents + data[:, i]*(1.0-percents)

            if not exist_nan:
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
            # 查找第一个非 NaN数据
            next_i = None
            for i in range(data.shape[1]):
                if not np.isnan(data[:, i]).any() and  not (data[:, i]==9999.0).any() :
                    next_i=i+1
                    break
            if not next_i is None:
                self.b_first = False
                self.X_prev = data[:, next_i-1]
                self.P_prev = np.zeros_like(self.X_prev)
            else:
                next_i = data.shape[1]+1
        else:
            if np.isnan(data[:, 0]).any() or (data[:, 0]==9999.0).any():
                data[:, 0] = self.X_prev
            self.K_prev = self.P_prev / (self.P_prev + self.R)
            data[:, 0] = self.X_prev + self.K_prev * (data[:, 0] - self.X_prev)
            self.P_prev = self.P_prev - self.K_prev * self.P_prev + self.Q
            next_i = 1
        for i in range(next_i, data.shape[1]):
            if np.isnan(data[:, i]).any() or (data[:, i]==9999.0).any():
                data[:, i] = self.X_prev
            K = self.P_prev / (self.P_prev + self.R)
            data[:, i] = data[:, i-1] + K * (data[:, i] - data[:, i-1])
            P = self.P_prev - K * self.P_prev + self.Q
            self.P_prev = P
            self.K_prev = K
            self.X_prev = data[:, i]


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
        self.data_first_len = int(order*2)
        self.data_first = np.zeros((self.data_first_len,), dtype=np.float32)
        self.prev_val = None
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
        if self.prev_val is None:
            self.prev_val = np.zeros((data.shape[0],))
        index = self.index
        for chl, data_chl in enumerate(data): 
            for i in range(len(data_chl)):                
                if np.isnan(data_chl[i]) or data_chl[i]==9999.0:
                    data_chl[i] = self.prev_val[chl]
                z, self.zi = lfilter(self.b, self.a, [data_chl[i]], zi=self.zi)
                #if index+i<self.order:
                #    continue
                if index+i<self.data_first_len:
                    self.data_first[index+i] = data_chl[i]
                    if index+i<3:
                        _data = self.data_first[:index+i+1]
                    else:
                        _data = self.data_first[index+i-3:index+i+1]
                    data_chl[i] = _data.mean()
                else:
                    data_chl[i] = z
                self.prev_val[chl]=data_chl[i]
        #self.index += data.shape[0]
        self.index += data.shape[1]
    

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
                 feature=None, binding_obj=None, filter_type=0, q_size=4, std_th=0.05, percent=0.8, Q=1e-6, R=4e-4, fs=5., cutoff=1., order=5):
        '''
        扩展属性
        -----
        @param feature     - [list like] 目标特征向量
        @param binding_obj - [int or object] 绑定对象
        @param filter_type - [Track] exts2滤波器类型
        @param q_size      - [Track] 队列长度
        @param std_th      - [Track] 相对误差域值
        @param percent     - [Track] 奇异点保留前置能量比，当设为1.0即为完全用前置点替换奇异点
        @param Q           - [Track] 卡尔曼滤波器参数
        @param R           - [Track] 卡尔曼滤波器参数
        @param fs          - [Track] 采样率
        @param cutoff      - [Track] 截止频率, Hz
        @param order       - [Track] 滤波器阶数
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
        self.exts2_prev = None  # 上一帧扩展信息
        self.exts2      = None  # 当前帧扩展信息
        self.t_prev     = None  # 上一帧时间(秒)
        self.t          = None  # 当前帧时间(秒)


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

    def get_exts2_prev(self):
        return np.array(self.exts2_prev)

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

    def update(self, kf, detection, save_to=None):
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
        if False and not save_to is None:
            save_file_origin_mean = '%s/mean/%s-%d/origin.txt' % (save_to, detection.flag, self.track_id)
            save_file_origin_ext2 = '%s/exts2/%s-%d/origin.txt' % (save_to, detection.flag, self.track_id)
            os.makedirs(os.path.dirname(save_file_origin_mean), exist_ok=True)
            os.makedirs(os.path.dirname(save_file_origin_ext2), exist_ok=True)
            with open(save_file_origin_mean, 'a+') as f:
                f.write(str(list(detection.to_xyah()))[1:-1])
                f.write('\n')
            with open(save_file_origin_ext2, 'a+') as f:
                f.write(str(list(detection.exts2))[1:-1])
                f.write('\n')

        if not save_to is None:
            save_file = '%s/%s_%d/origin.txt' % (save_to, detection.flag, self.track_id)
            os.makedirs(os.path.dirname(save_file), exist_ok=True)
            with open(save_file, 'a+') as f:
                f.write(str(list(np.r_[self.to_tlbr(), detection.exts2]))[1:-1])
                f.write('\n')

        # 跟踪/滤波处理
        # ------------
        self.mean, self.covariance = kf.update(
            self.mean, self.covariance, detection.to_xyah())
        self.features.append(detection.feature)
        self.objFilter.filter(detection)

        if not self.exts2 is None:
            self.exts2_prev = self.exts2.copy()
        self.t_prev = self.t

        self.exts2 = detection.exts2
        self.t = detection.t


        # 数据采集: 滤波后
        # ---------------
        if False and not save_to is None:
            t, l, b, r = self.to_tlbr()
            t, l, b, r = int(t), int(l), int(b), int(r)
            #save_file_filter_mean = '%s/mean/%s-%d-%d-%d-%d-%d/filter.txt' % (save_to, detection.flag, self.track_id, t, l, b, r)
            #save_file_filter_ext2 = '%s/exts2/%s-%d-%d-%d-%d-%d/filter.txt' % (save_to, detection.flag, self.track_id, t, l, b, r)
            save_file_filter_mean = '%s/mean/%s-%d/filter.txt' % (save_to, detection.flag, self.track_id)
            save_file_filter_ext2 = '%s/exts2/%s-%d/filter.txt' % (save_to, detection.flag, self.track_id)
            os.makedirs(os.path.dirname(save_file_filter_mean), exist_ok=True)
            os.makedirs(os.path.dirname(save_file_filter_ext2), exist_ok=True)
            #print('save_file_filter_mean: ', save_file_filter_mean)
            #print('save_file_filter_ext2: ', save_file_filter_ext2)
            with open(save_file_filter_mean, 'a+') as f:
                f.write(str(list(self.mean))[1:-1])
                f.write('\n')
            with open(save_file_filter_ext2, 'a+') as f:
                f.write(str(list(self.exts2))[1:-1])
                f.write('\n')

        if not save_to is None:
            save_file = '%s/%s_%d/filter.txt' % (save_to, detection.flag, self.track_id)
            os.makedirs(os.path.dirname(save_file), exist_ok=True)
            with open(save_file, 'a+') as f:
                f.write(str(list(np.r_[self.to_tlbr(), detection.exts2]))[1:-1])
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
