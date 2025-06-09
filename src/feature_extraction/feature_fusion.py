"""
特征融合模块
将图像特征和文本特征融合为统一的特征表示
"""
from pathlib import Path
import numpy as np
from typing import Dict, Tuple, List
from sklearn.preprocessing import StandardScaler

def normalize_features(features: np.ndarray) -> np.ndarray:
    """对特征进行标准化"""
    scaler = StandardScaler()
    return scaler.fit_transform(features)

def fuse_features(
    image_features: Dict[str, Dict[str, np.ndarray]],
    text_features: Dict[str, Dict[str, np.ndarray]],
    output_dir: Path = None
) -> Dict[str, np.ndarray]:
    """融合图像和文本特征
    
    Args:
        image_features: 图像特征，格式为 {类别: {图像路径: 特征}}
        text_features: 文本特征，格式为 {类别: {文本内容: 特征}}
        output_dir: 特征保存目录，如果为None则不保存
    
    Returns:
        Dict[str, np.ndarray]: {类别: 融合特征矩阵}
    """
    # 收集所有特征
    all_features = []
    all_categories = []
    
    # 对每个类别的每个样本进行特征融合
    for category in image_features.keys():
        if category not in text_features:
            print(f"警告：类别 {category} 没有对应的文本特征，跳过")
            continue
            
        # 获取该类别的文本特征
        category_text_features = text_features[category]
        if not category_text_features:
            print(f"警告：类别 {category} 的文本特征为空，跳过")
            continue
            
        # 对该类别的每个图像样本进行处理
        for img_path, img_feature in image_features[category].items():
            # 对每个图像特征，都使用其类别对应的所有文本特征进行融合
            for text_feature in category_text_features.values():
                # 融合单个样本的特征
                fused_feature = np.concatenate([img_feature, text_feature])
                all_features.append(fused_feature)
                all_categories.append(category)
    
    if not all_features:
        raise ValueError("没有找到可以融合的特征！")
    
    # 将所有特征转换为数组
    all_features = np.array(all_features)
    
    # 对特征进行标准化
    scaler = StandardScaler()
    normalized_features = scaler.fit_transform(all_features)
    
    # 按类别组织特征
    fused_features = {}
    for feature, category in zip(normalized_features, all_categories):
        if category not in fused_features:
            fused_features[category] = []
        fused_features[category].append(feature)
    
    # 将列表转换为数组
    for category in fused_features:
        fused_features[category] = np.array(fused_features[category])
    
    # 保存融合特征
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        for category, features in fused_features.items():
            output_path = output_dir / f"{category}_fused_features.npy"
            np.save(output_path, features)
    
    return fused_features

def create_feature_dataset(
    data_dir: Path,
    output_dir: Path = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """创建用于训练的特征数据集
    
    Args:
        data_dir: 数据目录
        output_dir: 特征保存目录
    
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: (特征矩阵, 标签, 手写标记)
    """
    from .image_features import extract_image_features
    from .text_features import extract_text_features
    
    print("\n=== 特征提取过程 ===")
    # 1. 提取特征
    print("\n提取图像特征...")
    image_features = extract_image_features(data_dir, output_dir)
    print(f"图像特征类别数量: {len(image_features)}")
    print("图像特征样本数量:")
    for category, features in image_features.items():
        print(f"  {category}: {len(features)}")
    
    print("\n提取文本特征...")
    text_features = extract_text_features(data_dir, output_dir)
    print(f"文本特征类别数量: {len(text_features)}")
    print("文本特征样本数量:")
    for category, features in text_features.items():
        print(f"  {category}: {len(features)}")
    
    print("\n=== 特征融合过程 ===")
    # 2. 融合特征
    fused_features = fuse_features(image_features, text_features, output_dir)
    print(f"融合后的类别数量: {len(fused_features)}")
    print("融合后每个类别的样本数量:")
    for category, features in fused_features.items():
        print(f"  {category}: {len(features)}")
    
    print("\n=== 数据集创建过程 ===")
    # 3. 创建数据集
    X = []  # 特征
    y = []  # 标签
    handwritten = []  # 手写标记
    
    # 按类别排序，确保处理顺序一致
    sorted_categories = sorted(fused_features.keys())
    
    for category in sorted_categories:
        features = fused_features[category]
        # 从类别名称中提取主类别和手写标记
        # 类别名称格式：主类别_编号y 或 主类别_编号
        main_category = category.split('_')[0]  # 提取主类别（如"7"）
        is_handwritten = category.endswith('y')  # 检查是否为手写样本
        
        X.append(features)
        y.extend([main_category] * len(features))
        handwritten.extend([1 if is_handwritten else 0] * len(features))
    
    if not X:  # 如果没有数据
        raise ValueError("没有找到任何有效的特征数据！")
    
    X = np.vstack(X)
    y = np.array(y)
    handwritten = np.array(handwritten)
    
    print("\n=== 最终数据集统计 ===")
    print(f"特征矩阵形状: {X.shape}")
    print(f"标签数量: {len(y)}")
    print(f"手写标记数量: {len(handwritten)}")
    print(f"类别统计:")
    unique_labels = np.unique(y)
    for label in sorted(unique_labels):
        mask = y == label
        n_total = np.sum(mask)
        n_handwritten = np.sum(handwritten[mask] == 1)
        n_normal = np.sum(handwritten[mask] == 0)
        print(f"类别 {label}:")
        print(f"  - 总样本数: {n_total}")
        print(f"  - 手写样本数: {n_handwritten}")
        print(f"  - 非手写样本数: {n_normal}")
    
    # 4. 保存数据集
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        np.save(output_dir / "X.npy", X)
        np.save(output_dir / "y.npy", y)
        np.save(output_dir / "handwritten.npy", handwritten)
        print("\n数据集已保存到:", output_dir)
    
    return X, y, handwritten 