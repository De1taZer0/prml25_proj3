"""
聚类模块
实现多种聚类算法和评估
"""
import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import normalized_mutual_info_score, accuracy_score
from sklearn.metrics.cluster import contingency_matrix
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram
import seaborn as sns
from scipy.optimize import linear_sum_assignment

from pylab import mpl
# 设置显示中文字体
mpl.rcParams["font.sans-serif"] = ["SimHei"]
mpl.rcParams["axes.unicode_minus"] = False

class MultiClusterer:
    def __init__(self, random_state: int = 42):
        """初始化聚类器
        
        Args:
            random_state: 随机种子
        """
        self.random_state = random_state
        self.scaler = StandardScaler()
        
        # 聚类器字典
        self.clusterers = {
            'KMeans': None,
            'DBSCAN': None,
            'Hierarchical': None
        }
        
        # 聚类结果
        self.labels = {}
        self.metrics = {}
    
    def cluster_and_evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray = None,
        handwritten: np.ndarray = None,
        n_clusters: int = None,
        output_dir: Path = None
    ):
        """执行聚类并评估结果
        
        Args:
            X: 特征矩阵
            y: 真实标签（用于评估）
            handwritten: 手写标记
            n_clusters: 聚类数量
            output_dir: 结果保存目录
        """
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        # 数据预处理
        X_scaled = self.scaler.fit_transform(X)
        
        # 如果没有指定聚类数，且有真实标签，使用真实类别数
        if n_clusters is None and y is not None:
            n_clusters = len(np.unique(y))
        elif n_clusters is None:
            n_clusters = 10  # 默认值
        
        print(f"数据集大小: {X.shape}")
        print(f"聚类数量: {n_clusters}")
        
        # 1. K-Means聚类
        print("\n=== K-Means聚类 ===")
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=self.random_state,
            n_init=10
        )
        self.clusterers['KMeans'] = kmeans
        self.labels['KMeans'] = kmeans.fit_predict(X_scaled)
        
        # 2. DBSCAN聚类
        print("\n=== DBSCAN聚类 ===")
        # 根据数据分布自适应设置eps
        from sklearn.neighbors import NearestNeighbors
        neigh = NearestNeighbors(n_neighbors=2)
        nbrs = neigh.fit(X_scaled)
        distances, indices = nbrs.kneighbors(X_scaled)
        distances = np.sort(distances, axis=0)
        distances = distances[:,1]
        eps = np.percentile(distances, 90)  # 使用90%分位数作为eps
        
        dbscan = DBSCAN(eps=eps, min_samples=5)
        self.clusterers['DBSCAN'] = dbscan
        self.labels['DBSCAN'] = dbscan.fit_predict(X_scaled)
        
        # 3. 层次聚类
        print("\n=== 层次聚类 ===")
        hierarchical = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage='ward'
        )
        self.clusterers['Hierarchical'] = hierarchical
        self.labels['Hierarchical'] = hierarchical.fit_predict(X_scaled)
        
        # 评估结果
        if y is not None:
            self._evaluate_clustering(y)
        
        # 可视化结果
        if output_dir:
            self._visualize_results(X_scaled, handwritten, output_dir)
            
            # 保存结果
            results = {
                'labels': self.labels,
                'metrics': self.metrics
            }
            np.save(output_dir / 'clustering_results.npy', results)
    
    def _evaluate_clustering(self, y_true: np.ndarray):
        """评估聚类结果
        
        Args:
            y_true: 真实标签
        """
        self.metrics = {}
        
        for name, labels in self.labels.items():
            # 跳过DBSCAN的噪声点（标签为-1）
            if name == 'DBSCAN':
                mask = labels != -1
                if not np.any(mask):
                    print(f"{name}聚类未找到有效聚类，跳过评估")
                    continue
                labels = labels[mask]
                y_true_filtered = y_true[mask]
            else:
                y_true_filtered = y_true
            
            # 计算NMI
            nmi = normalized_mutual_info_score(y_true_filtered, labels)
            
            # 计算ACC
            acc = self._calculate_acc(y_true_filtered, labels)
            
            self.metrics[name] = {
                'NMI': nmi,
                'ACC': acc
            }
            
            print(f"\n{name}聚类结果:")
            print(f"NMI: {nmi:.4f}")
            print(f"ACC: {acc:.4f}")
            
            # 打印每个聚类的主要类别
            unique_clusters = np.unique(labels)
            print("\n各聚类的主要类别:")
            for cluster in unique_clusters:
                if cluster != -1:  # 跳过噪声点
                    cluster_mask = labels == cluster
                    cluster_true_labels = y_true_filtered[cluster_mask]
                    unique_true, counts = np.unique(cluster_true_labels, return_counts=True)
                    main_label = unique_true[np.argmax(counts)]
                    purity = np.max(counts) / len(cluster_true_labels)
                    print(f"聚类 {cluster}: 主要类别 {main_label} (纯度: {purity:.4f})")
    
    def _calculate_acc(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """计算聚类准确率
        
        Args:
            y_true: 真实标签
            y_pred: 预测标签
            
        Returns:
            聚类准确率
        """
        # 计算联列表（混淆矩阵）
        cm = contingency_matrix(y_true, y_pred)
        
        # 使用匈牙利算法找到最佳匹配
        row_ind, col_ind = linear_sum_assignment(-cm)
        
        # 计算最佳匹配下的准确率
        total = 0
        for i, j in zip(row_ind, col_ind):
            total += cm[i, j]
        
        return total / cm.sum()
    
    def _visualize_results(self, X: np.ndarray, handwritten: np.ndarray, output_dir: Path):
        """可视化聚类结果
        
        Args:
            X: 特征矩阵
            handwritten: 手写标记
            output_dir: 结果保存目录
        """
        # 1. 使用PCA降维到2D进行可视化
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        X_2d = pca.fit_transform(X)
        
        # 为每个聚类器创建散点图
        for name, labels in self.labels.items():
            plt.figure(figsize=(12, 6))
            
            # 如果是DBSCAN，处理噪声点
            if name == 'DBSCAN':
                # 绘制噪声点
                noise_mask = labels == -1
                plt.scatter(
                    X_2d[noise_mask, 0],
                    X_2d[noise_mask, 1],
                    c='gray',
                    marker='x',
                    label='噪声'
                )
                # 绘制聚类点
                plt.scatter(
                    X_2d[~noise_mask, 0],
                    X_2d[~noise_mask, 1],
                    c=labels[~noise_mask],
                    cmap='viridis'
                )
            else:
                plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap='viridis')
            
            plt.title(f'{name}聚类结果')
            plt.xlabel('第一主成分')
            plt.ylabel('第二主成分')
            plt.colorbar(label='聚类标签')
            if name == 'DBSCAN':
                plt.legend()
            plt.tight_layout()
            plt.savefig(output_dir / f'{name}_clusters.png')
            plt.close()
            
            # 如果有手写标记，创建手写vs非手写的可视化
            if handwritten is not None:
                plt.figure(figsize=(12, 6))
                
                # 分别绘制手写和非手写样本
                hw_mask = handwritten == 1
                non_hw_mask = handwritten == 0
                
                plt.scatter(
                    X_2d[hw_mask, 0],
                    X_2d[hw_mask, 1],
                    c=labels[hw_mask],
                    cmap='viridis',
                    marker='o',
                    label='手写'
                )
                plt.scatter(
                    X_2d[non_hw_mask, 0],
                    X_2d[non_hw_mask, 1],
                    c=labels[non_hw_mask],
                    cmap='viridis',
                    marker='^',
                    label='非手写'
                )
                
                plt.title(f'{name}聚类结果 (手写vs非手写)')
                plt.xlabel('第一主成分')
                plt.ylabel('第二主成分')
                plt.colorbar(label='聚类标签')
                plt.legend()
                plt.tight_layout()
                plt.savefig(output_dir / f'{name}_handwritten.png')
                plt.close()
        
        # 3. 为层次聚类创建树状图
        if 'Hierarchical' in self.clusterers:
            # 创建一个新的层次聚类器用于生成树状图
            linkage_matrix = AgglomerativeClustering(
                distance_threshold=0,
                n_clusters=None,
                linkage='ward',
                compute_distances=True
            ).fit(X)
            
            plt.figure(figsize=(15, 8))
            plt.title('层次聚类树状图')
            plt.xlabel('样本索引')
            plt.ylabel('距离')
            
            # 计算距离矩阵
            from scipy.cluster.hierarchy import linkage
            Z = linkage(X, method='ward')
            
            # 绘制树状图
            dendrogram(Z)
            plt.tight_layout()
            plt.savefig(output_dir / 'hierarchical_dendrogram.png')
            plt.close()
            
            # 保存层次聚类的合并历史
            np.save(output_dir / 'hierarchical_linkage.npy', Z) 