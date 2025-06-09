"""
运行分类实验
"""
from pathlib import Path
import numpy as np
from classification.classifier import MultiClassifier

def main():
    # 设置路径
    data_dir = Path("features")
    output_dir = Path("results/classification")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载数据
    print("\n=== 加载数据 ===")
    X = np.load(data_dir / "X.npy")
    y = np.load(data_dir / "Y.npy")
    handwritten = np.load(data_dir / "handwritten.npy")
    
    print(f"特征矩阵形状: {X.shape}")
    print(f"类别标签形状: {y.shape}")
    print(f"手写标记形状: {handwritten.shape}")
    
    # 创建分类器
    print("\n=== 开始分类实验 ===")
    classifier = MultiClassifier(random_state=42)
    
    for test_size in [0.2, 0.3, 0.4, 0.5]:
        # 训练并评估
        classifier.train_and_evaluate(
            X=X,
            y=y,
            handwritten=handwritten,
            test_size=test_size,
            output_dir=output_dir / f"test_size_{test_size}"
        )

if __name__ == "__main__":
    main() 