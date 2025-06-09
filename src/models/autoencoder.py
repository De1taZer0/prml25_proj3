"""
自编码器模型
用于特征降维和可视化
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

class Autoencoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 2):
        super().__init__()
        
        # 编码器
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
        
        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim)
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

class AutoencoderDimReduction:
    def __init__(self, latent_dim: int = 2, random_state: int = 42):
        self.latent_dim = latent_dim
        self.random_state = random_state
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """训练自编码器并返回降维后的特征
        
        Args:
            X: 输入特征矩阵
            
        Returns:
            降维后的特征矩阵
        """
        # 设置随机种子
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        
        # 创建模型
        input_dim = X.shape[1]
        self.model = Autoencoder(input_dim, self.latent_dim).to(self.device)
        
        # 准备数据
        X_tensor = torch.FloatTensor(X).to(self.device)
        dataset = TensorDataset(X_tensor)
        dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
        
        # 训练参数
        optimizer = optim.Adam(self.model.parameters())
        criterion = nn.MSELoss()
        
        # 训练模型
        self.model.train()
        print("\n训练自编码器...")
        for epoch in range(50):
            total_loss = 0
            for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/50", leave=False):
                x = batch[0]
                
                # 前向传播
                encoded, decoded = self.model(x)
                loss = criterion(decoded, x)
                
                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.6f}")
        
        # 获取降维结果
        self.model.eval()
        with torch.no_grad():
            encoded, _ = self.model(X_tensor)
            encoded = encoded.cpu().numpy()
        
        return encoded
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """对新数据进行降维
        
        Args:
            X: 输入特征矩阵
            
        Returns:
            降维后的特征矩阵
        """
        if self.model is None:
            raise ValueError("模型尚未训练，请先调用fit_transform")
        
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            encoded, _ = self.model(X_tensor)
            encoded = encoded.cpu().numpy()
        
        return encoded 