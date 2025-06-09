"""
集成分类器模块
包含投票和堆叠两种集成方式
"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression

class VotingClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, classifiers: dict, voting: str = 'hard'):
        """初始化投票分类器
        
        Args:
            classifiers: 分类器字典
            voting: 投票方式，'hard'表示硬投票，'soft'表示软投票
        """
        self.classifiers = classifiers
        self.voting = voting
        
    def fit(self, X: np.ndarray, y: np.ndarray):
        """训练所有基分类器
        
        Args:
            X: 特征矩阵
            y: 标签
        """
        # 训练每个基分类器
        for name, clf in self.classifiers.items():
            print(f"\n训练基分类器: {name}")
            clf.fit(X, y)
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测类别
        
        Args:
            X: 特征矩阵
            
        Returns:
            预测的类别
        """
        if self.voting == 'hard':
            # 硬投票：每个分类器投票，取多数
            predictions = []
            for clf in self.classifiers.values():
                predictions.append(clf.predict(X))
            predictions = np.array(predictions)
            return np.apply_along_axis(
                lambda x: np.bincount(x).argmax(),
                axis=0,
                arr=predictions
            )
        else:
            # 软投票：基于概率加权
            return self.predict_proba(X).argmax(axis=1)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """预测概率
        
        Args:
            X: 特征矩阵
            
        Returns:
            预测的概率
        """
        # 收集所有分类器的概率预测
        probas = []
        for clf in self.classifiers.values():
            probas.append(clf.predict_proba(X))
        
        # 平均所有概率
        return np.mean(probas, axis=0)

class StackingClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, classifiers: dict, meta_classifier=None, n_folds: int = 5, random_state: int = 42):
        """初始化堆叠分类器
        
        Args:
            classifiers: 基分类器字典
            meta_classifier: 元分类器，默认使用逻辑回归
            n_folds: 交叉验证折数
            random_state: 随机种子
        """
        self.classifiers = classifiers
        self.meta_classifier = meta_classifier or LogisticRegression(random_state=random_state)
        self.n_folds = n_folds
        self.random_state = random_state
        self.trained_classifiers = {}
        
    def fit(self, X: np.ndarray, y: np.ndarray):
        """训练堆叠分类器
        
        Args:
            X: 特征矩阵
            y: 标签
        """
        # 使用交叉验证生成元特征
        kfold = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        meta_features = np.zeros((X.shape[0], len(self.classifiers) * len(np.unique(y))))
        
        # 对每个基分类器
        for i, (name, clf) in enumerate(self.classifiers.items()):
            print(f"\n训练基分类器: {name}")
            # 创建一个新的分类器实例
            self.trained_classifiers[name] = clf.__class__(**clf.get_params())
            
            # 使用交叉验证生成元特征
            for train_idx, val_idx in kfold.split(X, y):
                # 在训练集上训练
                clf.fit(X[train_idx], y[train_idx])
                # 在验证集上预测概率
                proba = clf.predict_proba(X[val_idx])
                meta_features[val_idx, i*proba.shape[1]:(i+1)*proba.shape[1]] = proba
            
            # 在全部数据上训练最终模型
            self.trained_classifiers[name].fit(X, y)
        
        # 训练元分类器
        print("\n训练元分类器...")
        self.meta_classifier.fit(meta_features, y)
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测类别
        
        Args:
            X: 特征矩阵
            
        Returns:
            预测的类别
        """
        meta_features = self._get_meta_features(X)
        return self.meta_classifier.predict(meta_features)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """预测概率
        
        Args:
            X: 特征矩阵
            
        Returns:
            预测的概率
        """
        meta_features = self._get_meta_features(X)
        return self.meta_classifier.predict_proba(meta_features)
    
    def _get_meta_features(self, X: np.ndarray) -> np.ndarray:
        """生成元特征
        
        Args:
            X: 特征矩阵
            
        Returns:
            元特征矩阵
        """
        meta_features = []
        
        # 收集每个基分类器的预测概率
        for clf in self.trained_classifiers.values():
            meta_features.append(clf.predict_proba(X))
        
        # 将所有预测概率拼接在一起
        return np.hstack(meta_features) 