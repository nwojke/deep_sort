import numpy as np


class Detection:
    """
    表示单张图像中的边界框检测。

    Parameters
    ----------
    tlwh : array_like
        边界框格式 `(x, y, w, h)`.
    confidence : float
        检测器置信度分数。
    feature : array_like
        描述图像中包含目标的特征向量。
    class_id : int
        目标类别ID，用于多类别跟踪。
    class_name : str
        目标类别名称。

    Attributes
    ----------
    tlwh : ndarray
        边界框格式 `(top left x, top left y, width, height)`.
    confidence : float
        检测器置信度分数。
    feature : ndarray | None
        描述图像中包含目标的特征向量。
    class_id : int
        目标类别ID。
    class_name : str
        目标类别名称。
    """

    def __init__(self, tlwh, confidence, feature, class_id=0, class_name=None):
        self.tlwh = np.asarray(tlwh, dtype=np.float64)
        self.confidence = float(confidence)
        self.feature = np.asarray(feature, dtype=np.float32) if feature is not None else None
        self.class_id = int(class_id)
        self.class_name = class_name

    def to_tlbr(self):
        """将边界框转换为格式 `(min x, min y, max x, max y)`,
        即 `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    def to_xyah(self):
        """将边界框转换为格式 `(center x, center y, aspect ratio, height)`,
        其中 aspect ratio 是 `width / height`.
        """
        ret = self.tlwh.copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret
