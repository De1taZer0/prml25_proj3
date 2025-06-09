"""
聚类结果分析脚本
"""
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from scipy.cluster.hierarchy import dendrogram

from pylab import mpl
# 设置显示中文字体
mpl.rcParams["font.sans-serif"] = ["SimHei"]

def analyze_cluster_distribution(labels, y_true, handwritten, output_dir):
    """分析聚类分布情况
    
    Args:
        labels: 聚类标签字典
        y_true: 真实标签
        handwritten: 手写标记
        output_dir: 结果保存目录
    """
    # 确保标签是数值类型
    y_true = y_true.astype(int)
    
    # 创建分析报告
    report_path = output_dir / 'cluster_analysis_report.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=== 聚类分析报告 ===\n\n")
        
        # 分析每种聚类方法的结果
        for method, cluster_labels in labels.items():
            f.write(f"\n{method}聚类分析:\n")
            f.write("-" * 50 + "\n")
            
            # 确保聚类标签是数值类型
            cluster_labels = cluster_labels.astype(int)
            
            # 1. 分析类别分布
            f.write("\n1. 聚类标签分布:\n")
            unique_clusters = np.unique(cluster_labels)
            for cluster in unique_clusters:
                if cluster == -1:
                    f.write(f"噪声点数量: {np.sum(cluster_labels == cluster)}\n")
                else:
                    cluster_mask = cluster_labels == cluster
                    f.write(f"\n聚类 {cluster}:\n")
                    f.write(f"样本数量: {np.sum(cluster_mask)}\n")
                    
                    # 分析真实标签分布
                    true_labels = y_true[cluster_mask]
                    unique_true, counts_true = np.unique(true_labels, return_counts=True)
                    f.write("真实标签分布:\n")
                    for label, count in zip(unique_true, counts_true):
                        f.write(f"  标签 {label}: {count} ({count/len(true_labels)*100:.2f}%)\n")
                    
                    # 分析手写vs非手写
                    hw_mask = handwritten[cluster_mask]
                    hw_count = np.sum(hw_mask == 1)
                    non_hw_count = np.sum(hw_mask == 0)
                    f.write(f"手写样本: {hw_count} ({hw_count/len(hw_mask)*100:.2f}%)\n")
                    f.write(f"非手写样本: {non_hw_count} ({non_hw_count/len(hw_mask)*100:.2f}%)\n")
            
            # 2. 分析手写样本的聚类情况
            f.write("\n2. 手写样本聚类分析:\n")
            hw_mask = handwritten == 1
            hw_clusters = cluster_labels[hw_mask]
            unique_hw_clusters, hw_counts = np.unique(hw_clusters, return_counts=True)
            f.write("\n手写样本在各聚类中的分布:\n")
            for cluster, count in zip(unique_hw_clusters, hw_counts):
                if cluster == -1:
                    f.write(f"噪声点: {count} 个手写样本\n")
                else:
                    total_in_cluster = np.sum(cluster_labels == cluster)
                    f.write(f"聚类 {cluster}: {count} 个手写样本 ")
                    f.write(f"(占该聚类的 {count/total_in_cluster*100:.2f}%)\n")
            
            # 3. 分析非手写样本的聚类情况
            f.write("\n3. 非手写样本聚类分析:\n")
            non_hw_mask = handwritten == 0
            non_hw_clusters = cluster_labels[non_hw_mask]
            unique_non_hw_clusters, non_hw_counts = np.unique(non_hw_clusters, return_counts=True)
            f.write("\n非手写样本在各聚类中的分布:\n")
            for cluster, count in zip(unique_non_hw_clusters, non_hw_counts):
                if cluster == -1:
                    f.write(f"噪声点: {count} 个非手写样本\n")
                else:
                    total_in_cluster = np.sum(cluster_labels == cluster)
                    f.write(f"聚类 {cluster}: {count} 个非手写样本 ")
                    f.write(f"(占该聚类的 {count/total_in_cluster*100:.2f}%)\n")
            
            # 4. 分析聚类的纯度
            f.write("\n4. 聚类纯度分析:\n")
            for cluster in unique_clusters:
                if cluster != -1:
                    cluster_mask = cluster_labels == cluster
                    true_labels = y_true[cluster_mask]
                    unique_true, counts_true = np.unique(true_labels, return_counts=True)
                    max_label_count = np.max(counts_true)
                    purity = max_label_count / len(true_labels)
                    f.write(f"\n聚类 {cluster} 的纯度: {purity:.4f}\n")
                    f.write(f"主要类别: {unique_true[np.argmax(counts_true)]}\n")
            
            # 5. 创建混淆矩阵热力图
            if method != 'DBSCAN':  # DBSCAN可能有噪声点，需要特殊处理
                plt.figure(figsize=(10, 8))
                cm = confusion_matrix(y_true, cluster_labels)
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                plt.title(f'{method}聚类与真实标签的混淆矩阵')
                plt.xlabel('聚类标签')
                plt.ylabel('真实标签')
                plt.tight_layout()
                plt.savefig(output_dir / f'{method}_confusion_matrix.png')
                plt.close()

def analyze_hierarchical_clustering(linkage_matrix, output_dir):
    """分析层次聚类结果
    
    Args:
        linkage_matrix: 层次聚类的连接矩阵
        output_dir: 结果保存目录
    """
    report_path = output_dir / 'hierarchical_analysis_report.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=== 层次聚类分析报告 ===\n\n")
        
        # 1. 分析合并历史
        f.write("1. 合并历史分析\n")
        f.write("-" * 50 + "\n\n")
        
        n_samples = linkage_matrix.shape[0] + 1
        f.write(f"样本总数: {n_samples}\n")
        f.write(f"合并步骤数: {linkage_matrix.shape[0]}\n\n")
        
        # 分析合并距离
        distances = linkage_matrix[:, 2]
        f.write("合并距离统计:\n")
        f.write(f"最小距离: {np.min(distances):.4f}\n")
        f.write(f"最大距离: {np.max(distances):.4f}\n")
        f.write(f"平均距离: {np.mean(distances):.4f}\n")
        f.write(f"距离标准差: {np.std(distances):.4f}\n\n")
        
        # 分析每次合并的样本数
        sizes = np.zeros(linkage_matrix.shape[0])
        for i, merge in enumerate(linkage_matrix):
            left, right = int(merge[0]), int(merge[1])
            if left < n_samples:
                left_size = 1
            else:
                left_size = sizes[left - n_samples]
            if right < n_samples:
                right_size = 1
            else:
                right_size = sizes[right - n_samples]
            sizes[i] = left_size + right_size
        
        f.write("合并大小统计:\n")
        f.write(f"最小合并大小: {np.min(sizes):.0f}\n")
        f.write(f"最大合并大小: {np.max(sizes):.0f}\n")
        f.write(f"平均合并大小: {np.mean(sizes):.2f}\n")
        f.write(f"合并大小标准差: {np.std(sizes):.2f}\n")

def main():
    """主函数"""
    # 加载聚类结果
    results_dir = Path('results/clustering')
    if not results_dir.exists():
        print("错误：未找到聚类结果目录")
        return
    
    # 加载数据
    print("加载数据...")
    clustering_results = np.load(results_dir / 'clustering_results.npy', allow_pickle=True).item()
    data_dir = Path('features')
    y = np.load(data_dir / 'y.npy')
    handwritten = np.load(data_dir / 'handwritten.npy')
    
    # 分析聚类分布
    print("\n分析聚类分布...")
    analyze_cluster_distribution(
        labels=clustering_results['labels'],
        y_true=y,
        handwritten=handwritten,
        output_dir=results_dir
    )
    
    # 分析层次聚类结果
    print("\n分析层次聚类结果...")
    try:
        linkage_matrix = np.load(results_dir / 'hierarchical_linkage.npy')
        analyze_hierarchical_clustering(
            linkage_matrix=linkage_matrix,
            output_dir=results_dir
        )
    except Exception as e:
        print(f"警告：无法加载或分析层次聚类结果 - {e}")
    
    print("\n分析完成！请查看results/clustering目录下的分析报告。")

if __name__ == '__main__':
    main() 