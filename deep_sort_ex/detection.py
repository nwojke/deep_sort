# vim: expandtab:ts=4:sw=4
import numpy as np


class Detection(object):
    """
    This class represents a bounding box detection in a single image.

    Parameters
    ----------
    tlwh : array_like
        Bounding box in format `(x, y, w, h)`.
    confidence : float
        Detector confidence score.
    feature : array_like
        A feature vector that describes the object contained in this image.

    Attributes
    ----------
    tlwh : ndarray
        Bounding box in format `(top left x, top left y, width, height)`.
    confidence : ndarray
        Detector confidence score.
    feature : ndarray | NoneType
        A feature vector that describes the object contained in this image.

    """

    def __init__(self, tlwh, confidence, feature, t=None, exts1=None, exts2=None, binding_obj=None, flag=None):
        '''
        @param tlwh         - bbox: top, left, width, height
        @param confidence   - 目标检测置信度
        @param ffeature     - 目标图像特征码
        @param t            - 检测时间(秒)
        @param exts1        - 扩展属性: 扩展卡尔曼滤波器的向量(mean)， 需要与 tracker,track的n_extend参数配合使用
        @param exts2        - 扩展属性: 独立于卡尔曼滤波器，对扩展通道单独处理
        @param binding_obj  - 绑定原始目标检测的序号(或结构体对象)，方便数据源跟踪
        @param flag         - 跟踪目标标记， 如用于目标类别，由外部解析
        '''
        self.tlwh = np.asarray(tlwh, dtype=np.float)
        self.t = t 
        self.exts1 = exts1
        self.exts2 = exts2
        self.binding_obj = binding_obj
        self.flag = flag
        self.confidence = float(confidence)
        self.feature = np.asarray(feature, dtype=np.float32)

    def to_tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    def to_xyah(self):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = self.tlwh.copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        if self.exts1 is None:
            return ret
        else:
            return np.hstack([ret, self.exts1])
