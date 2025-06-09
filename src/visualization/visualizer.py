"""
可视化模块
使用多种降维方法可视化特征
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler, LabelEncoder
from models.autoencoder import AutoencoderDimReduction
import seaborn as sns

class FeatureVisualizer:
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
        # 初始化降维方法
        self.dim_reduction_methods = {
            'PCA': PCA(n_components=2, random_state=random_state),
            'LDA': LDA(n_components=2),
            't-SNE': TSNE(n_components=2, random_state=random_state),
            'Autoencoder': AutoencoderDimReduction(latent_dim=2, random_state=random_state)
        }
    
    def visualize_features(self, X: np.ndarray, y: np.ndarray, handwritten: np.ndarray, output_dir: Path = None):
        """使用多种方法可视化特征
        
        Args:
            X: 特征矩阵
            y: 类别标签
            handwritten: 手写标记
            output_dir: 输出目录
        """
        # 数据标准化
        X_scaled = self.scaler.fit_transform(X)
        
        # 将类别标签转换为数值类型
        y = np.array(y, dtype=int)
        
        # 为每种降维方法创建一个子图
        fig, axes = plt.subplots(2, 2, figsize=(20, 20))
        axes = axes.ravel()
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        
        for i, (method_name, method) in enumerate(self.dim_reduction_methods.items()):
            print(f"\n使用{method_name}进行降维...")
            # 降维
            if method_name == 'LDA':
                X_reduced = method.fit_transform(X_scaled, y)
            else:
                X_reduced = method.fit_transform(X_scaled)
            
            # 分别绘制手写和非手写样本
            for hw in [0, 1]:
                mask = handwritten == hw
                scatter = axes[i].scatter(
                    X_reduced[mask, 0],
                    X_reduced[mask, 1],
                    c=y[mask],
                    cmap='tab10',
                    marker='o' if hw == 0 else '^',
                    alpha=0.6,
                    label=f"{'手写' if hw == 1 else '非手写'}"
                )
            
            axes[i].set_title(f"{method_name}降维结果")
            axes[i].legend()
            
            # 添加颜色条
            cbar = plt.colorbar(scatter, ax=axes[i])
            cbar.set_label('类别')
            # 设置颜色条的刻度标签
            unique_labels = np.unique(y)
            cbar.set_ticks(unique_labels)
            cbar.set_ticklabels([str(label) for label in unique_labels])
        
        plt.tight_layout()
        
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_dir / "feature_visualization.png", dpi=300, bbox_inches='tight')
        
        plt.close()
        
        # 打印每种方法的方差解释率（如果有的话）
        print("\n=== 降维效果分析 ===")
        for method_name, method in self.dim_reduction_methods.items():
            if method_name == 'PCA' and hasattr(method, 'explained_variance_ratio_'):
                ratio = method.explained_variance_ratio_
                print(f"\nPCA方差解释率:")
                print(f"  第一维度: {ratio[0]:.4f}")
                print(f"  第二维度: {ratio[1]:.4f}")
                print(f"  累计: {sum(ratio):.4f}")
            elif method_name == 'LDA':
                print(f"\nLDA特征值比例:")
                if hasattr(method, 'explained_variance_ratio_'):
                    ratio = method.explained_variance_ratio_
                    print(f"  第一维度: {ratio[0]:.4f}")
                    print(f"  第二维度: {ratio[1]:.4f}")
                    print(f"  累计: {sum(ratio):.4f}")
            elif method_name == 't-SNE':
                print("\nt-SNE没有方差解释率指标")
            elif method_name == 'Autoencoder':
                print("\nAutoencoder没有方差解释率指标") 