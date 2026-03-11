import torch
import torch.nn as nn
import torchvision.transforms as T
import numpy as np
import cv2


class BasicBlock(nn.Module):
    """ResNet Basic Block。"""

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class DeepFeatureExtractor(nn.Module):
    """
    基于ResNet的行人重识别特征提取器。
    输入: 128x64 的行人图像
    输出: 128维特征向量
    """

    def __init__(self, num_classes=751, reid_dim=128):
        super(DeepFeatureExtractor, self).__init__()
        self.reid_dim = reid_dim

        # 初始卷积层
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNet层
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=1)

        # 全局平均池化
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # 特征投影层
        self.fc = nn.Linear(512, reid_dim)
        self.bn_fc = nn.BatchNorm1d(reid_dim)

        # 分类器（仅在训练时使用）
        self.classifier = nn.Linear(reid_dim, num_classes)

        self._initialize_weights()

    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(BasicBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        x = self.fc(x)
        x = self.bn_fc(x)
        x = nn.functional.normalize(x, p=2, dim=1)

        return x

    def extract_features(self, x):
        """提取特征（推理时使用）。"""
        return self.forward(x)


class FeatureExtractor:
    """
    特征提取器封装类，处理预处理和后处理。
    """

    def __init__(self, model_path=None, device='cuda', image_size=(128, 64)):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.image_size = image_size

        # 创建模型
        self.model = DeepFeatureExtractor(reid_dim=128)

        # 加载预训练权重（如果提供）
        if model_path is not None:
            self.load_weights(model_path)

        self.model.to(self.device)
        self.model.eval()

        # 图像预处理
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize(image_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def load_weights(self, model_path):
        """加载模型权重。"""
        checkpoint = torch.load(model_path, map_location=self.device)
        if 'state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        print(f"Loaded weights from {model_path}")

    def save_weights(self, model_path):
        """保存模型权重。"""
        torch.save(self.model.state_dict(), model_path)
        print(f"Saved weights to {model_path}")

    def _extract_image_patch(self, image, bbox):
        """从边界框提取图像块。"""
        x, y, w, h = bbox.astype(int)

        # 调整边界框以匹配目标宽高比
        target_aspect = self.image_size[1] / self.image_size[0]
        new_w = target_aspect * h
        x = int(x - (new_w - w) / 2)
        w = int(new_w)

        # 裁剪并调整大小
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(image.shape[1], x + w)
        y2 = min(image.shape[0], y + h)

        if x1 >= x2 or y1 >= y2:
            return None

        patch = image[y1:y2, x1:x2]
        patch = cv2.resize(patch, (self.image_size[1], self.image_size[0]))

        return patch

    def __call__(self, image, boxes):
        """
        提取图像中边界框的特征。

        Parameters
        ----------
        image : ndarray
            BGR格式的图像。
        boxes : ndarray
            边界框数组，格式为 (x, y, w, h)。

        Returns
        -------
        ndarray
            特征矩阵，每行对应一个边界框的特征。
        """
        if len(boxes) == 0:
            return np.array([])

        patches = []
        for box in boxes:
            patch = self._extract_image_patch(image, box)
            if patch is None:
                # 如果提取失败，使用随机噪声
                patch = np.random.randint(0, 256, (*self.image_size, 3), dtype=np.uint8)
            patches.append(patch)

        # 转换为张量
        patches_tensor = torch.stack([
            self.transform(patch) for patch in patches
        ]).to(self.device)

        # 提取特征
        with torch.no_grad():
            features = self.model.extract_features(patches_tensor)

        return features.cpu().numpy()


def create_box_encoder(model_path=None, device='cuda', batch_size=32):
    """
    创建边界框编码器。

    Parameters
    ----------
    model_path : str, optional
        模型权重路径。
    device : str
        计算设备。
    batch_size : int
        批处理大小。

    Returns
    -------
    callable
        编码器函数。
    """
    extractor = FeatureExtractor(model_path=model_path, device=device)

    def encoder(image, boxes):
        if len(boxes) == 0:
            return np.array([])

        features = []
        for i in range(0, len(boxes), batch_size):
            batch_boxes = boxes[i:i + batch_size]
            batch_features = extractor(image, batch_boxes)
            features.append(batch_features)

        return np.vstack(features) if features else np.array([])

    return encoder
