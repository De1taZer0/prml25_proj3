"""
可视化特征
"""
from pathlib import Path
import numpy as np
from visualization.visualizer import FeatureVisualizer

def main():
    # 加载数据
    data_dir = Path("features")
    X = np.load(data_dir / "X.npy")
    y = np.load(data_dir / "y.npy")
    handwritten = np.load(data_dir / "handwritten.npy")
    
    print("\n=== 数据集信息 ===")
    print(f"特征维度: {X.shape}")
    print(f"类别数量: {len(np.unique(y))}")
    print(f"手写样本数量: {np.sum(handwritten == 1)}")
    print(f"非手写样本数量: {np.sum(handwritten == 0)}")
    
    # 创建可视化器
    visualizer = FeatureVisualizer(random_state=42)
    
    # 创建输出目录
    output_dir = Path("results/visualization")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 可视化特征
    print("\n=== 开始可视化 ===")
    visualizer.visualize_features(X, y, handwritten, output_dir)

if __name__ == "__main__":
    main() 