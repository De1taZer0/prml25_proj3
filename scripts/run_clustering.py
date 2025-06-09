"""
聚类分析脚本
"""
import numpy as np
from pathlib import Path
from clustering.clusterer import MultiClusterer

from pylab import mpl
# 设置显示中文字体
mpl.rcParams["font.sans-serif"] = ["SimHei"]

def main():
    """主函数"""
    # 创建结果目录
    output_dir = Path('results/clustering')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载数据
    print("加载数据...")
    data_dir = Path('features')
    X = np.load(data_dir / 'X.npy')
    y = np.load(data_dir / 'y.npy')
    handwritten = np.load(data_dir / 'handwritten.npy')
    
    print(f"数据集大小: {X.shape}")
    print(f"类别数量: {len(np.unique(y))}")
    print(f"手写样本数量: {np.sum(handwritten == 1)}")
    print(f"非手写样本数量: {np.sum(handwritten == 0)}")
    
    # 执行聚类
    print("\n开始聚类分析...")
    clusterer = MultiClusterer()
    clusterer.cluster_and_evaluate(
        X=X,
        y=y,
        handwritten=handwritten,
        output_dir=output_dir
    )
    
    print("\n聚类分析完成！请查看results/clustering目录下的结果。")

if __name__ == '__main__':
    main() 