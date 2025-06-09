"""
特征提取模块
"""
from .image_features import extract_image_features
from .text_features import extract_text_features
from .feature_fusion import fuse_features, create_feature_dataset

__all__ = ['extract_image_features', 'extract_text_features', 'fuse_features', 'create_feature_dataset'] 