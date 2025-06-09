"""
特征提取脚本
"""
from pathlib import Path
from feature_extraction import create_feature_dataset
from utils import data_augmented_dir

def main():
    # 特征保存目录
    feature_dir = Path("features")
    feature_dir.mkdir(exist_ok=True)
    
    # 从增强后的数据中提取特征
    print("开始提取特征...")
    X, y, handwritten = create_feature_dataset(data_augmented_dir, feature_dir)
    print(f"特征提取完成！")
    print(f"特征维度: {X.shape}")
    print(f"样本数量: {len(y)}")
    print(f"类别数量: {len(set(y))}")
    print(f"手写样本数量: {sum(handwritten == 1)}")
    print(f"非手写样本数量: {sum(handwritten == 0)}")

if __name__ == "__main__":
    main() 