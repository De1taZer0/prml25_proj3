"""
图像特征提取
使用预训练的ResNet50模型提取图像特征
"""
from pathlib import Path
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
from tqdm import tqdm
import os
from typing import Dict, List, Tuple

class ImageFeatureExtractor:
    def __init__(self, device: str = None):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # 加载预训练的ResNet50模型
        self.model = models.resnet50(pretrained=True)
        # 移除最后的全连接层，只保留特征提取部分
        self.model = nn.Sequential(*list(self.model.children())[:-1])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # 图像预处理
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    @torch.no_grad()
    def extract_single(self, image_path: Path) -> np.ndarray:
        """提取单个图像的特征"""
        # 读取并预处理图像
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image).unsqueeze(0)
        image = image.to(self.device)
        
        # 提取特征
        features = self.model(image)
        features = features.squeeze().cpu().numpy()
        
        return features
    
    def extract_batch(self, image_paths: List[Path], batch_size: int = 32) -> Dict[Path, np.ndarray]:
        """批量提取图像特征"""
        features = {}
        
        for i in tqdm(range(0, len(image_paths), batch_size), desc="提取图像特征"):
            batch_paths = image_paths[i:i + batch_size]
            batch_images = []
            
            for path in batch_paths:
                image = Image.open(path).convert('RGB')
                image = self.transform(image)
                batch_images.append(image)
            
            # 将批次图像转换为tensor
            batch_tensor = torch.stack(batch_images).to(self.device)
            
            # 提取特征
            with torch.no_grad():
                batch_features = self.model(batch_tensor)
                batch_features = batch_features.squeeze().cpu().numpy()
            
            # 如果只有一张图片，需要保持维度
            if len(batch_paths) == 1:
                batch_features = batch_features.reshape(1, -1)
            
            # 保存特征
            for path, feature in zip(batch_paths, batch_features):
                features[path] = feature
        
        return features

def extract_image_features(data_dir: Path, output_dir: Path = None) -> Dict[str, Dict[str, np.ndarray]]:
    """提取数据集中所有图像的特征
    
    Args:
        data_dir: 数据目录，包含多个类别子目录
        output_dir: 特征保存目录，如果为None则不保存
    
    Returns:
        Dict[str, Dict[str, np.ndarray]]: {类别: {图像路径: 特征}}
    """
    # 初始化特征提取器
    extractor = ImageFeatureExtractor()
    
    # 收集所有图像路径
    image_paths = []
    categories = []
    
    for dir_path in data_dir.iterdir():
        if dir_path.is_dir():
            # 使用完整的目录名作为类别名
            category = dir_path.name
            
            # 处理图像
            formula_path = dir_path / 'formula.png'
            if formula_path.exists():
                image_paths.append(formula_path)
                categories.append(category)
    
    print(f"找到的图像总数: {len(image_paths)}")
    print(f"其中手写图像数量: {sum(1 for c in categories if c.endswith('y'))}")
    print(f"非手写图像数量: {sum(1 for c in categories if not c.endswith('y'))}")
    
    # 批量提取特征
    features = extractor.extract_batch(image_paths)
    
    # 按类别组织特征
    category_features = {}
    for path, category in zip(image_paths, categories):
        if category not in category_features:
            category_features[category] = {}
        category_features[category][str(path)] = features[path]
    
    # 保存特征
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        for category, category_dict in category_features.items():
            output_path = output_dir / f"{category}_image_features.npy"
            np.save(output_path, category_dict)
    
    return category_features 