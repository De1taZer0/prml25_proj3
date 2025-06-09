"""
CNN分类器模型
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from sklearn.base import BaseEstimator, ClassifierMixin

class CNNClassifier(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        
        # 将特征重塑为2D图像形式
        # 假设特征可以重塑为正方形
        self.side_length = int(np.sqrt(input_dim))
        self.reshape_dim = self.side_length ** 2
        if self.reshape_dim < input_dim:
            self.side_length += 1
            self.reshape_dim = self.side_length ** 2
        
        # CNN层
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))  # 自适应池化到1x1
        )
        
        # 全连接层
        self.fc_layers = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        
        # Softmax层
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        # 将输入重塑为图像形式
        batch_size = x.size(0)
        x_padded = torch.zeros(batch_size, self.reshape_dim).to(x.device)
        x_padded[:, :x.size(1)] = x
        x = x_padded.view(batch_size, 1, self.side_length, self.side_length)
        
        # 通过CNN层
        x = self.conv_layers(x)
        x = x.view(batch_size, -1)
        
        # 通过全连接层
        x = self.fc_layers(x)
        
        return x
    
    def predict_proba(self, x):
        return self.softmax(self.forward(x))

class CNNClassifierWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, input_dim: int = None, num_classes: int = None, random_state: int = 42):
        """初始化CNN分类器
        
        Args:
            input_dim: 输入特征维度
            num_classes: 类别数量
            random_state: 随机种子
        """
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.random_state = random_state
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.criterion = None
        self.optimizer = None
    
    def _initialize_model(self):
        """初始化模型和训练参数"""
        if self.input_dim is None or self.num_classes is None:
            raise ValueError("input_dim和num_classes必须在fit之前设置")
        
        # 设置随机种子
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        
        # 创建模型
        self.model = CNNClassifier(self.input_dim, self.num_classes).to(self.device)
        
        # 训练参数
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters())
    
    def get_params(self, deep=True):
        """获取参数，实现scikit-learn接口"""
        return {
            'input_dim': self.input_dim,
            'num_classes': self.num_classes,
            'random_state': self.random_state
        }
    
    def set_params(self, **parameters):
        """设置参数，实现scikit-learn接口"""
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """训练模型
        
        Args:
            X: 特征矩阵
            y: 标签
        """
        # 如果模型未初始化，设置输入维度和类别数
        if self.input_dim is None:
            self.input_dim = X.shape[1]
        if self.num_classes is None:
            self.num_classes = len(np.unique(y))
        
        # 初始化模型
        self._initialize_model()
        
        # 准备数据
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.LongTensor(y).to(self.device)
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        # 训练模型
        self.model.train()
        print("\n训练CNN...")
        for epoch in range(100):
            total_loss = 0
            for batch_X, batch_y in tqdm(dataloader, desc=f"Epoch {epoch+1}/100", leave=False):
                # 前向传播
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                
                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.6f}")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测类别
        
        Args:
            X: 特征矩阵
            
        Returns:
            预测的类别
        """
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            outputs = self.model(X_tensor)
            _, predicted = torch.max(outputs, 1)
            return predicted.cpu().numpy()
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """预测概率
        
        Args:
            X: 特征矩阵
            
        Returns:
            预测的概率
        """
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            proba = self.model.predict_proba(X_tensor)
            return proba.cpu().numpy() 