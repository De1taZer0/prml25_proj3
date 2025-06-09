"""
文本特征提取
使用预训练的BERT模型提取文本特征
"""
from pathlib import Path
import torch
from transformers import BertModel, BertTokenizer
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Tuple

class TextFeatureExtractor:
    def __init__(self, device: str = None):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # 加载预训练的BERT模型和分词器
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        self.model = BertModel.from_pretrained('bert-base-chinese')
        self.model = self.model.to(self.device)
        self.model.eval()
    
    @torch.no_grad()
    def extract_single(self, text: str) -> np.ndarray:
        """提取单个文本的特征"""
        # 对文本进行分词和编码
        inputs = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # 提取特征
        outputs = self.model(**inputs)
        # 使用[CLS]标记的输出作为文本特征
        features = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        
        return features.squeeze()
    
    def extract_batch(self, texts: List[str], batch_size: int = 32) -> Dict[str, np.ndarray]:
        """批量提取文本特征"""
        features = {}
        
        for i in tqdm(range(0, len(texts), batch_size), desc="提取文本特征"):
            batch_texts = texts[i:i + batch_size]
            
            # 对文本进行分词和编码
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # 提取特征
            with torch.no_grad():
                outputs = self.model(**inputs)
                batch_features = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            
            # 保存特征
            for text, feature in zip(batch_texts, batch_features):
                features[text] = feature
        
        return features

def extract_text_features(data_dir: Path, output_dir: Path = None) -> Dict[str, Dict[str, np.ndarray]]:
    """提取数据集中所有文本的特征
    
    Args:
        data_dir: 数据目录，包含多个类别子目录
        output_dir: 特征保存目录，如果为None则不保存
    
    Returns:
        Dict[str, Dict[str, np.ndarray]]: {类别: {文本内容: 特征}}
    """
    # 初始化特征提取器
    extractor = TextFeatureExtractor()
    
    # 收集所有文本
    texts = []
    text_paths = []
    categories = []
    
    for dir_path in data_dir.iterdir():
        if dir_path.is_dir():
            # 使用完整的目录名作为类别名
            category = dir_path.name
            
            # 处理文本
            description_path = dir_path / 'description.txt'
            if description_path.exists():
                with open(description_path, 'r', encoding='utf-8') as f:
                    text = f.read().strip()
                    texts.append(text)
                    text_paths.append(description_path)
                    categories.append(category)
    
    print(f"找到的文本总数: {len(texts)}")
    print(f"其中手写样本对应的文本数量: {sum(1 for c in categories if c.endswith('y'))}")
    print(f"非手写样本对应的文本数量: {sum(1 for c in categories if not c.endswith('y'))}")
    
    # 批量提取特征
    features = extractor.extract_batch(texts)
    
    # 按类别组织特征
    category_features = {}
    for text, category in zip(texts, categories):
        if category not in category_features:
            category_features[category] = {}
        category_features[category][text] = features[text]
    
    # 保存特征
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        for category, category_dict in category_features.items():
            output_path = output_dir / f"{category}_text_features.npy"
            np.save(output_path, category_dict)
    
    return category_features 