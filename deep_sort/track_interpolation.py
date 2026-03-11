import numpy as np
from scipy.interpolate import interp1d


class TrackInterpolator:
    """
    轨迹插值器，用于填充被遮挡期间的缺失帧。

    当一个目标被遮挡若干帧后又被找回时，使用插值算法填补中间缺失的帧。
    支持线性插值和样条插值。

    Parameters
    ----------
    max_gap : int
        最大可插值的帧间隔。超过此值的间隙不会被插值。
    min_track_length : int
        执行插值所需的最小轨迹长度。
    interpolation_method : str
        插值方法，可选 'linear' 或 'cubic'。
    """

    def __init__(self, max_gap=30, min_track_length=5, interpolation_method='linear'):
        self.max_gap = max_gap
        self.min_track_length = min_track_length
        self.interpolation_method = interpolation_method

    def interpolate_track(self, track_history):
        """
        对单个轨迹进行插值。

        Parameters
        ----------
        track_history : list
            轨迹历史记录，每个元素包含帧索引和状态信息。

        Returns
        -------
        list
            插值后的轨迹点列表。
        """
        if len(track_history) < self.min_track_length:
            return track_history

        # 提取帧索引和边界框信息
        frames = []
        bboxes = []

        for state in track_history:
            # 从mean中提取边界框 (x, y, a, h)
            mean = state['mean']
            x, y, a, h = mean[0], mean[1], mean[2], mean[3]
            # 转换为xywh
            w = a * h
            x_tl = x - w / 2
            y_tl = y - h / 2

            frames.append(state.get('frame_idx', len(frames)))
            bboxes.append([x_tl, y_tl, w, h])

        frames = np.array(frames)
        bboxes = np.array(bboxes)

        # 检测间隙
        gaps = np.diff(frames)
        interpolated_states = []

        current_idx = 0
        interpolated_states.append(track_history[0])

        for i, gap in enumerate(gaps):
            if gap > 1 and gap <= self.max_gap:
                # 需要插值
                start_frame = frames[i]
                end_frame = frames[i + 1]
                start_bbox = bboxes[i]
                end_bbox = bboxes[i + 1]

                # 创建插值点
                interp_frames = np.arange(start_frame + 1, end_frame)

                # 对每个维度进行插值
                interp_bboxes = np.zeros((len(interp_frames), 4))
                for dim in range(4):
                    if self.interpolation_method == 'linear':
                        f = interp1d([start_frame, end_frame],
                                    [start_bbox[dim], end_bbox[dim]],
                                    kind='linear')
                    else:  # cubic
                        f = interp1d([start_frame, end_frame],
                                    [start_bbox[dim], end_bbox[dim]],
                                    kind='cubic')
                    interp_bboxes[:, dim] = f(interp_frames)

                # 创建插值状态
                for j, (f_idx, bbox) in enumerate(zip(interp_frames, interp_bboxes)):
                    x_tl, y_tl, w, h = bbox
                    x = x_tl + w / 2
                    y = y_tl + h / 2
                    a = w / h if h > 0 else 1.0

                    # 创建新的状态
                    interp_state = {
                        'mean': np.array([x, y, a, h, 0, 0, 0, 0]),
                        'covariance': track_history[i]['covariance'].copy(),
                        'frame_idx': int(f_idx),
                        'interpolated': True
                    }
                    interpolated_states.append(interp_state)

            interpolated_states.append(track_history[i + 1])
            current_idx = i + 1

        return interpolated_states

    def interpolate_tracks(self, tracks):
        """
        对多个轨迹进行插值。

        Parameters
        ----------
        tracks : list
            轨迹列表，每个轨迹是一个历史状态列表。

        Returns
        -------
        list
            插值后的轨迹列表。
        """
        interpolated_tracks = []
        for track in tracks:
            interpolated_track = self.interpolate_track(track)
            interpolated_tracks.append(interpolated_track)
        return interpolated_tracks


class TrajectorySmoother:
    """
    轨迹平滑器，使用卡尔曼平滑或移动平均来平滑轨迹。

    Parameters
    ----------
    window_size : int
        移动平均窗口大小。
    method : str
        平滑方法，可选 'moving_average' 或 'savgol'。
    """

    def __init__(self, window_size=5, method='moving_average'):
        self.window_size = window_size
        self.method = method

    def smooth(self, track_history):
        """
        平滑轨迹。

        Parameters
        ----------
        track_history : list
            轨迹历史记录。

        Returns
        -------
        list
            平滑后的轨迹。
        """
        if len(track_history) < self.window_size:
            return track_history

        # 提取边界框
        bboxes = []
        for state in track_history:
            mean = state['mean']
            x, y, a, h = mean[0], mean[1], mean[2], mean[3]
            w = a * h
            x_tl = x - w / 2
            y_tl = y - h / 2
            bboxes.append([x_tl, y_tl, w, h])

        bboxes = np.array(bboxes)

        # 应用平滑
        if self.method == 'moving_average':
            smoothed_bboxes = self._moving_average(bboxes)
        elif self.method == 'savgol':
            smoothed_bboxes = self._savgol_filter(bboxes)
        else:
            smoothed_bboxes = bboxes

        # 更新状态
        smoothed_history = []
        for i, (state, bbox) in enumerate(zip(track_history, smoothed_bboxes)):
            x_tl, y_tl, w, h = bbox
            x = x_tl + w / 2
            y = y_tl + h / 2
            a = w / h if h > 0 else 1.0

            smoothed_state = state.copy()
            smoothed_state['mean'] = np.array([x, y, a, h,
                                               state['mean'][4], state['mean'][5],
                                               state['mean'][6], state['mean'][7]])
            smoothed_history.append(smoothed_state)

        return smoothed_history

    def _moving_average(self, data):
        """应用移动平均。"""
        smoothed = np.copy(data)
        half_window = self.window_size // 2

        for i in range(len(data)):
            start = max(0, i - half_window)
            end = min(len(data), i + half_window + 1)
            smoothed[i] = np.mean(data[start:end], axis=0)

        return smoothed

    def _savgol_filter(self, data):
        """应用Savitzky-Golay滤波器。"""
        from scipy.signal import savgol_filter

        smoothed = np.copy(data)
        for dim in range(data.shape[1]):
            smoothed[:, dim] = savgol_filter(
                data[:, dim],
                window_length=min(self.window_size, len(data) // 2 * 2 + 1),
                polyorder=2
            )
        return smoothed


class TrackPostProcessor:
    """
    轨迹后处理器，整合插值和平滑功能。

    Parameters
    ----------
    enable_interpolation : bool
        是否启用轨迹插值。
    enable_smoothing : bool
        是否启用轨迹平滑。
    max_gap : int
        最大插值间隔。
    smooth_window : int
        平滑窗口大小。
    """

    def __init__(self, enable_interpolation=True, enable_smoothing=True,
                 max_gap=30, smooth_window=5):
        self.enable_interpolation = enable_interpolation
        self.enable_smoothing = enable_smoothing

        self.interpolator = TrackInterpolator(max_gap=max_gap) if enable_interpolation else None
        self.smoother = TrajectorySmoother(window_size=smooth_window) if enable_smoothing else None

    def process(self, tracks):
        """
        处理轨迹。

        Parameters
        ----------
        tracks : list
            轨迹列表。

        Returns
        -------
        list
            处理后的轨迹列表。
        """
        processed_tracks = []

        for track in tracks:
            processed_track = track

            # 插值
            if self.interpolator is not None:
                processed_track = self.interpolator.interpolate_track(processed_track)

            # 平滑
            if self.smoother is not None:
                processed_track = self.smoother.smooth(processed_track)

            processed_tracks.append(processed_track)

        return processed_tracks

    def process_frame_results(self, track_history_dict):
        """
        处理帧级别的跟踪结果。

        Parameters
        ----------
        track_history_dict : dict
            轨迹历史字典，键为track_id，值为历史状态列表。

        Returns
        -------
        dict
            处理后的轨迹字典。
        """
        tracks = list(track_history_dict.values())
        processed_tracks = self.process(tracks)

        return {
            track_id: processed_track
            for track_id, processed_track in zip(track_history_dict.keys(), processed_tracks)
        }
