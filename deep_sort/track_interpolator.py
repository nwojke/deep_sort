import numpy as np
from scipy.interpolate import interp1d
from deep_sort.track import Track

class TrackInterpolator:
    """
    轨迹插值与断点平滑处理类
    """
    
    def __init__(self, max_gap=30):
        """
        初始化轨迹插值器
        
        参数
        ----------
        max_gap : int
            最大允许的轨迹中断帧数
        """
        self.max_gap = max_gap
        self.track_history = {}
    
    def add_track(self, track_id, frame_id, bbox, class_id):
        """
        添加轨迹点
        
        参数
        ----------
        track_id : int
            轨迹ID
        frame_id : int
            帧ID
        bbox : ndarray
            边界框 (tlwh格式)
        class_id : int
            类别ID
        """
        if track_id not in self.track_history:
            self.track_history[track_id] = []
        
        self.track_history[track_id].append((frame_id, bbox, class_id))
    
    def offline_interpolate_tracks(self):
        """
        离线对所有轨迹进行插值处理，使用scipy进行平滑插值
        
        返回
        ----------
        dict
            插值后的轨迹历史
        """
        interpolated_history = {}
        
        for track_id, track_points in self.track_history.items():
            if len(track_points) < 2:
                interpolated_history[track_id] = track_points
                continue
            
            # 按帧ID排序
            sorted_points = sorted(track_points, key=lambda x: x[0])
            
            # 提取帧ID和边界框坐标
            frame_ids = [p[0] for p in sorted_points]
            bboxes = [p[1] for p in sorted_points]
            class_ids = [p[2] for p in sorted_points]
            
            # 将边界框转换为numpy数组
            bboxes = np.array(bboxes)
            
            # 计算帧ID之间的间隙
            gaps = np.diff(frame_ids)
            
            # 构建插值后的轨迹
            interpolated = [sorted_points[0]]
            
            for i in range(1, len(sorted_points)):
                prev_frame = frame_ids[i-1]
                curr_frame = frame_ids[i]
                gap = gaps[i-1]
                
                # 如果间隙过大，不进行插值
                if gap > self.max_gap:
                    interpolated.append(sorted_points[i])
                    continue
                
                # 对每个边界框坐标进行单独插值
                prev_bbox = bboxes[i-1]
                curr_bbox = bboxes[i]
                
                # 创建插值函数（使用线性插值）
                interp_functions = []
                for j in range(4):  # tlwh四个坐标
                    interp_func = interp1d([prev_frame, curr_frame], [prev_bbox[j], curr_bbox[j]], kind='linear')
                    interp_functions.append(interp_func)
                
                # 生成中间帧的插值结果
                for j in range(1, gap):
                    interp_frame = prev_frame + j
                    interp_bbox = np.array([f(interp_frame) for f in interp_functions])
                    interpolated.append((interp_frame, interp_bbox, class_ids[i-1]))
                
                interpolated.append(sorted_points[i])
            
            interpolated_history[track_id] = interpolated
        
        return interpolated_history
    
    def save_interpolated_tracks(self, output_file):
        """
        保存插值后的轨迹到文件
        
        参数
        ----------
        output_file : str
            输出文件路径
        """
        interpolated_history = self.offline_interpolate_tracks()
        np.save(output_file, interpolated_history)
    
    def load_interpolated_tracks(self, input_file):
        """
        从文件加载插值后的轨迹
        
        参数
        ----------
        input_file : str
            输入文件路径
        """
        self.track_history = np.load(input_file, allow_pickle=True).item()
    
    def get_interpolated_track(self, track_id):
        """
        获取指定轨迹的插值结果
        
        参数
        ----------
        track_id : int
            轨迹ID
        
        返回
        ----------
        list
            插值后的轨迹点列表
        """
        interpolated_history = self.offline_interpolate_tracks()
        return interpolated_history.get(track_id, [])