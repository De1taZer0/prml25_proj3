"""
分类器性能分析脚本
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import KFold
from classification.classifier import MultiClassifier
from sklearn.metrics import accuracy_score

from pylab import mpl
# 设置显示中文字体
mpl.rcParams["font.sans-serif"] = ["SimHei"]

def plot_split_ratio_impact(X, y, handwritten, test_sizes=[0.2, 0.3, 0.4, 0.5], n_trials=5):
    """分析测试集比例对分类性能的影响
    
    Args:
        X: 特征矩阵
        y: 标签
        handwritten: 手写标记
        test_sizes: 测试集比例列表
        n_trials: 每个比例的重复次数
    """
    # 记录每个分类器在不同划分比例下的性能
    results = {
        'KNN': [], 'NaiveBayes': [], 'DecisionTree': [], 
        'SVM': [], 'CNN': [], 'Voting(Hard)': [], 
        'Voting(Soft)': [], 'Stacking': []
    }
    std_results = {name: [] for name in results.keys()}
    
    for test_size in test_sizes:
        print(f"\n测试集比例: {test_size}")
        accuracies = {name: [] for name in results.keys()}
        
        for trial in range(n_trials):
            print(f"Trial {trial + 1}/{n_trials}")
            classifier = MultiClassifier()
            classifier.train_and_evaluate(X, y, handwritten, test_size=test_size)
            
            # 收集每个分类器的准确率
            for name in classifier.classifiers.keys():
                accuracies[name].append(classifier.evaluator.accuracy)
        
        # 计算平均值和标准差
        for name in results.keys():
            results[name].append(np.mean(accuracies[name]))
            std_results[name].append(np.std(accuracies[name]))
    
    # 绘制结果
    plt.figure(figsize=(12, 6))
    for name in results.keys():
        plt.errorbar(test_sizes, results[name], yerr=std_results[name], 
                    label=name, marker='o', capsize=5)
    
    plt.xlabel('测试集比例')
    plt.ylabel('准确率')
    plt.title('测试集比例对分类性能的影响')
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('results/split_ratio_impact.png')
    plt.close()
    
    return results, std_results

def analyze_handwritten_impact(X, y, handwritten):
    """分析手写样本对分类性能的影响
    
    Args:
        X: 特征矩阵
        y: 标签
        handwritten: 手写标记
    """
    # 分别获取手写和非手写样本的索引
    hw_idx = np.where(handwritten == 1)[0]
    non_hw_idx = np.where(handwritten == 0)[0]
    
    # 创建结果目录
    results_dir = Path('results/handwritten_analysis')
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # 分别评估手写和非手写样本
    print("\n评估手写样本性能...")
    hw_classifier = MultiClassifier()
    hw_classifier.train_and_evaluate(
        X[hw_idx], y[hw_idx], 
        output_dir=results_dir/'handwritten'
    )
    
    print("\n评估非手写样本性能...")
    non_hw_classifier = MultiClassifier()
    non_hw_classifier.train_and_evaluate(
        X[non_hw_idx], y[non_hw_idx],
        output_dir=results_dir/'non_handwritten'
    )
    
    # 获取准确率
    hw_acc = []
    non_hw_acc = []
    classifier_names = list(hw_classifier.classifiers.keys())
    
    for name in classifier_names:
        hw_acc.append(hw_classifier.evaluator.accuracy)
        non_hw_acc.append(non_hw_classifier.evaluator.accuracy)
    
    # 绘制对比图
    plt.figure(figsize=(12, 6))
    x = np.arange(len(classifier_names))
    width = 0.35
    
    plt.bar(x - width/2, hw_acc, width, label='手写样本')
    plt.bar(x + width/2, non_hw_acc, width, label='非手写样本')
    
    plt.xlabel('分类器')
    plt.ylabel('准确率')
    plt.title('手写vs非手写样本的分类性能对比')
    plt.xticks(x, classifier_names, rotation=45)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('results/handwritten_comparison.png')
    plt.close()
    
    return hw_acc, non_hw_acc

def analyze_stability(X, y, handwritten, n_splits=5, n_trials=5):
    """分析分类器的稳定性
    
    Args:
        X: 特征矩阵
        y: 标签
        handwritten: 手写标记
        n_splits: 交叉验证折数
        n_trials: 重复次数
    """
    # 记录每个分类器在不同划分下的性能
    results = {
        'KNN': [], 'NaiveBayes': [], 'DecisionTree': [], 
        'SVM': [], 'CNN': [], 'Voting(Hard)': [], 
        'Voting(Soft)': [], 'Stacking': []
    }
    
    for trial in range(n_trials):
        print(f"\nTrial {trial + 1}/{n_trials}")
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=trial)
        
        for fold, (train_idx, test_idx) in enumerate(kf.split(X), 1):
            print(f"Fold {fold}/{n_splits}")
            
            # 准备数据
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            hw_train = handwritten[train_idx] if handwritten is not None else None
            hw_test = handwritten[test_idx] if handwritten is not None else None
            
            # 训练和评估
            classifier = MultiClassifier()
            classifier.train_and_evaluate(X_train, y_train, hw_train)
            
            # 收集结果
            for name in classifier.classifiers.keys():
                results[name].append(classifier.evaluator.accuracy)
    
    # 计算每个分类器的平均性能和标准差
    stats = {}
    for name, accuracies in results.items():
        stats[name] = {
            'mean': np.mean(accuracies),
            'std': np.std(accuracies),
            'min': np.min(accuracies),
            'max': np.max(accuracies)
        }
    
    # 绘制箱线图
    plt.figure(figsize=(12, 6))
    plt.boxplot([results[name] for name in results.keys()], 
                labels=list(results.keys()))
    plt.xlabel('分类器')
    plt.ylabel('准确率')
    plt.title('分类器性能稳定性分析')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('results/classifier_stability.png')
    plt.close()
    
    return stats

def analyze_results(test_size: float, results_dir: Path):
    """分析分类结果
    
    Args:
        test_size: 测试集比例
        results_dir: 结果目录路径
    """
    # 加载测试数据
    print(f"\n分析测试集比例 {test_size} 的结果...")
    exp_dir = results_dir / 'classification' / f'test_size_{test_size}'
    data_dir = exp_dir / 'data'
    if not data_dir.exists():
        print(f"错误：未找到测试数据目录 {data_dir}")
        return None
    
    try:
        y_test = np.load(data_dir / 'y_test.npy')
        hw_test = np.load(data_dir / 'hw_test.npy') if (data_dir / 'hw_test.npy').exists() else None
    except Exception as e:
        print(f"错误：加载测试数据失败 - {e}")
        return None
    
    # 加载分类结果
    try:
        results = np.load(exp_dir / 'results' / 'results.npy', allow_pickle=True).item()
    except Exception as e:
        print(f"错误：加载分类结果失败 - {e}")
        return None
    
    # 1. 分析分类器性能
    print("\n=== 分类器性能分析 ===")
    print("-" * 50)
    
    performance = {}
    for name, metrics in results['metrics'].items():
        print(f"\n{name}:")
        performance[name] = {
            'accuracy': metrics['overall']['accuracy'],
            'precision': metrics['overall']['precision'],
            'recall': metrics['overall']['recall'],
            'f1': metrics['overall']['f1']
        }
        print(f"准确率: {metrics['overall']['accuracy']:.4f}")
        print(f"精确率: {metrics['overall']['precision']:.4f}")
        print(f"召回率: {metrics['overall']['recall']:.4f}")
        print(f"F1分数: {metrics['overall']['f1']:.4f}")
        if 'mean_auc' in metrics['overall']:
            performance[name]['auc'] = metrics['overall']['mean_auc']
            print(f"平均AUC: {metrics['overall']['mean_auc']:.4f}")
    
    # 2. 分析手写样本的影响
    if hw_test is not None:
        print("\n=== 手写样本影响分析 ===")
        print("-" * 50)
        
        hw_acc = []
        non_hw_acc = []
        classifier_names = list(results['predictions'].keys())
        
        for name in classifier_names:
            y_pred = results['predictions'][name]
            
            # 计算手写和非手写样本的准确率
            hw_mask = hw_test == 1
            non_hw_mask = hw_test == 0
            
            if np.any(hw_mask):
                hw_acc.append(accuracy_score(y_test[hw_mask], y_pred[hw_mask]))
                performance[name]['hw_accuracy'] = hw_acc[-1]
            if np.any(non_hw_mask):
                non_hw_acc.append(accuracy_score(y_test[non_hw_mask], y_pred[non_hw_mask]))
                performance[name]['non_hw_accuracy'] = non_hw_acc[-1]
        
        # 绘制对比图
        plt.figure(figsize=(12, 6))
        x = np.arange(len(classifier_names))
        width = 0.35
        
        plt.bar(x - width/2, hw_acc, width, label='手写样本')
        plt.bar(x + width/2, non_hw_acc, width, label='非手写样本')
        
        plt.xlabel('分类器')
        plt.ylabel('准确率')
        plt.title(f'手写vs非手写样本的分类性能对比 (测试集比例: {test_size})')
        plt.xticks(x, classifier_names, rotation=45)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(exp_dir / 'handwritten_comparison.png')
        plt.close()
        
        # 打印详细结果
        print("\n各分类器在手写和非手写样本上的表现:")
        for i, name in enumerate(classifier_names):
            print(f"\n{name}:")
            print(f"手写样本准确率: {hw_acc[i]:.4f}")
            print(f"非手写样本准确率: {non_hw_acc[i]:.4f}")
            print(f"性能差异: {abs(hw_acc[i] - non_hw_acc[i]):.4f}")
    
    # 3. 分析分类器稳定性
    print("\n=== 分类器稳定性分析 ===")
    print("-" * 50)
    
    for name, metrics in results['metrics'].items():
        print(f"\n{name}:")
        overall_metrics = metrics['overall']
        print(f"准确率: {overall_metrics['accuracy']:.4f}")
        if 'handwritten' in metrics:
            hw_metrics = metrics['handwritten']
            if 'handwritten' in hw_metrics:
                print(f"手写样本准确率: {hw_metrics['handwritten']['accuracy']:.4f}")
            if 'non_handwritten' in hw_metrics:
                print(f"非手写样本准确率: {hw_metrics['non_handwritten']['accuracy']:.4f}")
    
    # 4. 生成分析报告
    with open(exp_dir / 'analysis_report.txt', 'w', encoding='utf-8') as f:
        f.write(f"=== 分类分析报告 (测试集比例: {test_size}) ===\n\n")
        
        # 1. 总体性能
        f.write("1. 分类器总体性能\n")
        f.write("-" * 50 + "\n")
        for name, metrics in results['metrics'].items():
            f.write(f"\n{name}:\n")
            overall = metrics['overall']
            f.write(f"准确率: {overall['accuracy']:.4f}\n")
            f.write(f"精确率: {overall['precision']:.4f}\n")
            f.write(f"召回率: {overall['recall']:.4f}\n")
            f.write(f"F1分数: {overall['f1']:.4f}\n")
            if 'mean_auc' in overall:
                f.write(f"平均AUC: {overall['mean_auc']:.4f}\n")
                f.write("\n各类别AUC:\n")
                for class_idx, auc_score in overall['class_auc'].items():
                    f.write(f"类别 {class_idx}: {auc_score:.4f}\n")
        
        # 2. 手写样本影响
        if hw_test is not None:
            f.write("\n\n2. 手写样本的影响\n")
            f.write("-" * 50 + "\n")
            for i, name in enumerate(classifier_names):
                f.write(f"\n{name}:\n")
                f.write(f"手写样本准确率: {hw_acc[i]:.4f}\n")
                f.write(f"非手写样本准确率: {non_hw_acc[i]:.4f}\n")
                f.write(f"性能差异: {abs(hw_acc[i] - non_hw_acc[i]):.4f}\n")
        
        # 3. 分类器稳定性
        f.write("\n\n3. 分类器稳定性分析\n")
        f.write("-" * 50 + "\n")
        for name, metrics in results['metrics'].items():
            f.write(f"\n{name}:\n")
            if 'handwritten' in metrics:
                hw_metrics = metrics['handwritten']
                if 'handwritten' in hw_metrics:
                    hw = hw_metrics['handwritten']
                    f.write(f"手写样本性能:\n")
                    f.write(f"  准确率: {hw['accuracy']:.4f}\n")
                    f.write(f"  精确率: {hw['precision']:.4f}\n")
                    f.write(f"  召回率: {hw['recall']:.4f}\n")
                    f.write(f"  F1分数: {hw['f1']:.4f}\n")
                if 'non_handwritten' in hw_metrics:
                    non_hw = hw_metrics['non_handwritten']
                    f.write(f"非手写样本性能:\n")
                    f.write(f"  准确率: {non_hw['accuracy']:.4f}\n")
                    f.write(f"  精确率: {non_hw['precision']:.4f}\n")
                    f.write(f"  召回率: {non_hw['recall']:.4f}\n")
                    f.write(f"  F1分数: {non_hw['f1']:.4f}\n")
    
    return performance

def plot_test_size_impact(performances):
    """绘制测试集比例对分类性能的影响
    
    Args:
        performances: 不同测试集比例下的性能指标
    """
    results_dir = Path('results/classification')
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # 准备数据
    test_sizes = sorted(performances.keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    if 'auc' in list(performances[test_sizes[0]].values())[0]:
        metrics.append('auc')
    
    # 为每个指标创建一个图
    for metric in metrics:
        plt.figure(figsize=(12, 6))
        
        for name in performances[test_sizes[0]].keys():
            values = [performances[size][name][metric] for size in test_sizes]
            plt.plot(test_sizes, values, marker='o', label=name)
        
        plt.xlabel('测试集比例')
        plt.ylabel(metric.upper())
        plt.title(f'测试集比例对{metric.upper()}的影响')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(results_dir / f'test_size_impact_{metric}.png')
        plt.close()
    
    # 为手写和非手写样本创建单独的图
    if 'hw_accuracy' in list(performances[test_sizes[0]].values())[0]:
        plt.figure(figsize=(12, 6))
        
        for name in performances[test_sizes[0]].keys():
            hw_values = [performances[size][name]['hw_accuracy'] for size in test_sizes]
            non_hw_values = [performances[size][name]['non_hw_accuracy'] for size in test_sizes]
            
            plt.plot(test_sizes, hw_values, marker='o', linestyle='-', label=f'{name} (手写)')
            plt.plot(test_sizes, non_hw_values, marker='s', linestyle='--', label=f'{name} (非手写)')
        
        plt.xlabel('测试集比例')
        plt.ylabel('准确率')
        plt.title('测试集比例对手写和非手写样本分类性能的影响')
        plt.grid(True)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(results_dir / 'test_size_impact_handwritten.png')
        plt.close()

def main():
    """主函数"""
    # 加载分类结果
    results_dir = Path('results')
    if not results_dir.exists():
        print("错误：未找到结果目录")
        return
    
    # 分析不同测试集比例的结果
    test_sizes = [0.2, 0.3, 0.4, 0.5]
    performances = {}
    
    for test_size in test_sizes:
        print(f"\n分析测试集比例 {test_size} 的结果...")
        performance = analyze_results(test_size, results_dir)
        if performance is not None:
            performances[test_size] = performance
    
    # 绘制测试集比例影响图
    if performances:
        print("\n绘制测试集比例影响图...")
        plot_test_size_impact(performances)
        print("\n分析完成！请查看results/classification目录下的分析报告和可视化结果。")
    else:
        print("\n错误：没有找到有效的分类结果。")

if __name__ == '__main__':
    main() 