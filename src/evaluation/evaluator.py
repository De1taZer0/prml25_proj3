"""
分类器评估模块
包含评估指标计算和结果可视化
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    roc_auc_score
)
from sklearn.preprocessing import label_binarize
import seaborn as sns

class ClassifierEvaluator:
    def __init__(self, output_dir: Path = None):
        """初始化评估器
        
        Args:
            output_dir: 结果保存目录
        """
        self.output_dir = Path(output_dir) if output_dir else None
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray = None, handwritten: np.ndarray = None):
        """评估分类器性能
        
        Args:
            y_true: 真实标签
            y_pred: 预测标签
            y_prob: 预测概率（用于计算AUC）
            handwritten: 手写标记，用于分别评估手写和非手写样本
            
        Returns:
            包含各项评估指标的字典
        """
        metrics = {}
        
        # 计算总体性能
        print("\n=== 总体性能评估 ===")
        metrics['overall'] = self._evaluate_and_print(y_true, y_pred, y_prob)
        self._plot_confusion_matrix(y_true, y_pred, "总体混淆矩阵")
        if y_prob is not None:
            self._plot_roc_curves(y_true, y_prob, "总体ROC曲线")
        
        # 如果提供了手写标记，分别评估手写和非手写样本
        if handwritten is not None:
            metrics['handwritten'] = {}
            
            # 评估非手写样本
            mask_non_hw = handwritten == 0
            if np.any(mask_non_hw):
                print("\n=== 非手写样本性能评估 ===")
                metrics['handwritten']['non_handwritten'] = self._evaluate_and_print(
                    y_true[mask_non_hw], 
                    y_pred[mask_non_hw],
                    y_prob[mask_non_hw] if y_prob is not None else None
                )
                self._plot_confusion_matrix(
                    y_true[mask_non_hw], 
                    y_pred[mask_non_hw],
                    "非手写样本混淆矩阵"
                )
                if y_prob is not None:
                    self._plot_roc_curves(
                        y_true[mask_non_hw],
                        y_prob[mask_non_hw],
                        "非手写样本ROC曲线"
                    )
            
            # 评估手写样本
            mask_hw = handwritten == 1
            if np.any(mask_hw):
                print("\n=== 手写样本性能评估 ===")
                metrics['handwritten']['handwritten'] = self._evaluate_and_print(
                    y_true[mask_hw], 
                    y_pred[mask_hw],
                    y_prob[mask_hw] if y_prob is not None else None
                )
                self._plot_confusion_matrix(
                    y_true[mask_hw], 
                    y_pred[mask_hw],
                    "手写样本混淆矩阵"
                )
                if y_prob is not None:
                    self._plot_roc_curves(
                        y_true[mask_hw],
                        y_prob[mask_hw],
                        "手写样本ROC曲线"
                    )
        
        return metrics
    
    def _evaluate_and_print(self, y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray = None):
        """计算并打印评估指标
        
        Args:
            y_true: 真实标签
            y_pred: 预测标签
            y_prob: 预测概率
            
        Returns:
            包含各项评估指标的字典
        """
        metrics = {}
        
        # 计算各项指标
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, average='macro')
        metrics['recall'] = recall_score(y_true, y_pred, average='macro')
        metrics['f1'] = f1_score(y_true, y_pred, average='macro')
        
        # 打印总体指标
        print(f"准确率: {metrics['accuracy']:.4f}")
        print(f"宏平均精确率: {metrics['precision']:.4f}")
        print(f"宏平均召回率: {metrics['recall']:.4f}")
        print(f"宏平均F1分数: {metrics['f1']:.4f}")
        
        # 如果有概率预测，计算并打印AUC
        if y_prob is not None:
            # 获取类别列表
            classes = np.unique(y_true)
            n_classes = len(classes)
            
            # 将标签转换为one-hot编码
            y_true_bin = label_binarize(y_true, classes=classes)
            
            # 计算每个类别的AUC
            auc_scores = []
            for i in range(n_classes):
                if len(np.unique(y_true_bin[:, i])) > 1:  # 确保不是单一类别
                    auc_score = roc_auc_score(y_true_bin[:, i], y_prob[:, i])
                    auc_scores.append(auc_score)
            
            # 计算平均AUC
            metrics['mean_auc'] = np.mean(auc_scores)
            metrics['class_auc'] = {i: score for i, score in enumerate(auc_scores)}
            
            print(f"宏平均AUC: {metrics['mean_auc']:.4f}")
            print("\n各类别AUC:")
            for i, auc_score in enumerate(auc_scores):
                print(f"  类别 {classes[i]}: {auc_score:.4f}")
        
        # 打印详细分类报告
        print("\n分类报告:")
        print(classification_report(y_true, y_pred, digits=4))
        
        return metrics
    
    def _plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, title: str):
        """绘制混淆矩阵
        
        Args:
            y_true: 真实标签
            y_pred: 预测标签
            title: 图表标题
        """
        # 计算混淆矩阵
        cm = confusion_matrix(y_true, y_pred)
        
        # 创建图形
        plt.figure(figsize=(10, 8))
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 绘制热力图
        sns.heatmap(
            cm, 
            annot=True,  # 显示数值
            fmt='d',     # 整数格式
            cmap='Blues',  # 使用蓝色调色板
            xticklabels=sorted(np.unique(y_true)),  # 设置x轴标签
            yticklabels=sorted(np.unique(y_true))   # 设置y轴标签
        )
        
        # 设置标题和轴标签
        plt.title(title)
        plt.xlabel('预测标签')
        plt.ylabel('真实标签')
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图形
        if self.output_dir:
            # 将标题中的空格替换为下划线，作为文件名
            filename = f"confusion_matrix_{title.replace(' ', '_')}.png"
            plt.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight')
        
        plt.close()
    
    def _plot_roc_curves(self, y_true: np.ndarray, y_prob: np.ndarray, title: str):
        """绘制ROC曲线
        
        Args:
            y_true: 真实标签
            y_prob: 预测概率
            title: 图表标题
        """
        # 获取类别列表
        classes = np.unique(y_true)
        n_classes = len(classes)
        
        # 将标签转换为one-hot编码
        y_true_bin = label_binarize(y_true, classes=classes)
        
        # 创建图形
        plt.figure(figsize=(10, 8))
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 绘制每个类别的ROC曲线
        for i in range(n_classes):
            if len(np.unique(y_true_bin[:, i])) > 1:  # 确保不是单一类别
                fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
                roc_auc = auc(fpr, tpr)
                plt.plot(
                    fpr, tpr,
                    label=f'类别 {classes[i]} (AUC = {roc_auc:.2f})'
                )
        
        # 绘制随机猜测的基准线
        plt.plot([0, 1], [0, 1], 'k--', label='随机猜测')
        
        # 设置图形属性
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('假阳性率')
        plt.ylabel('真阳性率')
        plt.title(title)
        plt.legend(loc="lower right")
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图形
        if self.output_dir:
            # 将标题中的空格替换为下划线，作为文件名
            filename = f"roc_curves_{title.replace(' ', '_')}.png"
            plt.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight')
        
        plt.close() 