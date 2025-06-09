"""
分类器模块
包含多种分类算法的实现和评估
"""
from pathlib import Path
import numpy as np
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from evaluation.evaluator import ClassifierEvaluator
from models.cnn_classifier import CNNClassifierWrapper
from models.ensemble_classifier import VotingClassifier, StackingClassifier

class MultiClassifier:
    def __init__(self, random_state: int = 42):
        """初始化分类器
        
        Args:
            random_state: 随机种子
        """
        self.random_state = random_state
        self.num_classes = None
        self.input_dim = None
        self.label_encoder = LabelEncoder()
        
        # 初始化基分类器
        self.base_classifiers = {
            'KNN': KNeighborsClassifier(n_neighbors=5),
            'NaiveBayes': GaussianNB(),
            'DecisionTree': DecisionTreeClassifier(random_state=random_state),
            'SVM': SVC(probability=True, random_state=random_state),
            'CNN': None
        }
        
        # 初始化分类器字典（包括集成分类器）
        self.classifiers = self.base_classifiers.copy()
        
        self.evaluator = None
    
    def train_and_evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        handwritten: np.ndarray = None,
        test_size: float = 0.2,
        output_dir: Path = None
    ):
        """训练并评估多个分类器
        
        Args:
            X: 特征矩阵
            y: 类别标签
            handwritten: 手写标记
            test_size: 测试集比例
            output_dir: 结果保存目录
        """
        # 编码标签
        y = self.label_encoder.fit_transform(y)
        
        # 记录数据维度
        self.input_dim = X.shape[1]
        self.num_classes = len(np.unique(y))
        
        # 创建CNN分类器
        self.base_classifiers['CNN'] = CNNClassifierWrapper(
            input_dim=self.input_dim,
            num_classes=self.num_classes,
            random_state=self.random_state
        )
        
        # 更新分类器字典
        self.classifiers = self.base_classifiers.copy()
        
        # 添加集成分类器
        self.classifiers['Voting(Hard)'] = VotingClassifier(
            classifiers=self.base_classifiers.copy(),
            voting='hard'
        )
        self.classifiers['Voting(Soft)'] = VotingClassifier(
            classifiers=self.base_classifiers.copy(),
            voting='soft'
        )
        self.classifiers['Stacking'] = StackingClassifier(
            classifiers=self.base_classifiers.copy(),
            random_state=self.random_state
        )
        
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=self.random_state,
            stratify=y
        )
        
        # 如果有手写标记，也要相应划分
        hw_train = hw_test = None
        if handwritten is not None:
            _, hw_test = train_test_split(
                handwritten,
                test_size=test_size,
                random_state=self.random_state,
                stratify=y
            )
        
        # 创建结果目录
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            results_dir = output_dir / 'results'
            results_dir.mkdir(exist_ok=True)
            data_dir = output_dir / 'data'
            data_dir.mkdir(exist_ok=True)
            
            # 保存测试数据
            np.save(data_dir / 'X_test.npy', X_test)
            np.save(data_dir / 'y_test.npy', y_test)
            if hw_test is not None:
                np.save(data_dir / 'hw_test.npy', hw_test)
        
        # 训练并评估每个分类器
        results = {
            'predictions': {},
            'probabilities': {},
            'metrics': {}
        }
        
        for name, clf in self.classifiers.items():
            print(f"\n=== {name} ===")
            
            # 训练模型
            print(f"训练{name}...")
            clf.fit(X_train, y_train)
            
            # 预测
            print(f"预测中...")
            y_pred = clf.predict(X_test)
            y_prob = clf.predict_proba(X_test)
            
            # 保存预测结果
            results['predictions'][name] = y_pred
            results['probabilities'][name] = y_prob
            
            # 评估
            print(f"评估{name}性能...")
            if output_dir:
                clf_dir = results_dir / name
                clf_dir.mkdir(parents=True, exist_ok=True)
                
                # 保存预测结果
                np.save(clf_dir / 'predictions.npy', y_pred)
                np.save(clf_dir / 'probabilities.npy', y_prob)
                
                # 计算评估指标
                evaluator = ClassifierEvaluator(clf_dir)
                metrics = evaluator.evaluate(y_test, y_pred, y_prob, hw_test)
                results['metrics'][name] = metrics
        
        # 保存完整结果
        if output_dir:
            np.save(results_dir / 'results.npy', results)
    
    def predict(self, X: np.ndarray, classifier_name: str = 'Stacking') -> np.ndarray:
        """使用指定的分类器进行预测
        
        Args:
            X: 特征矩阵
            classifier_name: 分类器名称
            
        Returns:
            预测标签
        """
        if classifier_name not in self.classifiers:
            raise ValueError(f"未知的分类器: {classifier_name}")
        
        y_pred = self.classifiers[classifier_name].predict(X)
        return self.label_encoder.inverse_transform(y_pred)
    
    def predict_proba(self, X: np.ndarray, classifier_name: str = 'Stacking') -> np.ndarray:
        """使用指定的分类器进行概率预测
        
        Args:
            X: 特征矩阵
            classifier_name: 分类器名称
            
        Returns:
            预测概率
        """
        if classifier_name not in self.classifiers:
            raise ValueError(f"未知的分类器: {classifier_name}")
        
        return self.classifiers[classifier_name].predict_proba(X) 