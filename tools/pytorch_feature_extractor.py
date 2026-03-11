import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50
import cv2
import numpy as np

class PyTorchFeatureExtractor:
    def __init__(self, model_name='resnet50', feature_dim=128):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = resnet50(pretrained=True)
        # 替换最后一层以输出指定维度的特征
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, feature_dim)
        self.model.to(self.device)
        self.model.eval()
        
        # 图像预处理
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((128, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def extract_image_patch(self, image, bbox, patch_shape):
        bbox = np.array(bbox)
        if patch_shape is not None:
            target_aspect = float(patch_shape[1]) / patch_shape[0]
            new_width = target_aspect * bbox[3]
            bbox[0] -= (new_width - bbox[2]) / 2
            bbox[2] = new_width
        
        bbox[2:] += bbox[:2]
        bbox = bbox.astype(np.int64)
        
        bbox[:2] = np.maximum(0, bbox[:2])
        bbox[2:] = np.minimum(np.asarray(image.shape[:2][::-1]) - 1, bbox[2:])
        if np.any(bbox[:2] >= bbox[2:]):
            return None
        sx, sy, ex, ey = bbox
        image = image[sy:ey, sx:ex]
        image = cv2.resize(image, tuple(patch_shape[::-1]))
        return image
    
    def __call__(self, image, boxes):
        image_patches = []
        for box in boxes:
            patch = self.extract_image_patch(image, box, (128, 64))
            if patch is None:
                patch = np.random.uniform(0., 255., (128, 64, 3)).astype(np.uint8)
            image_patches.append(patch)
        
        if not image_patches:
            return np.array([])
        
        # 预处理图像
        batch = torch.stack([self.transform(patch) for patch in image_patches]).to(self.device)
        
        # 提取特征
        with torch.no_grad():
            features = self.model(batch).cpu().numpy()
        
        return features

def create_pytorch_box_encoder(model_name='resnet50', feature_dim=128):
    encoder = PyTorchFeatureExtractor(model_name, feature_dim)
    
    def encode(image, boxes):
        return encoder(image, boxes)
    
    return encode