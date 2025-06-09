from pathlib import Path
import random
from typing import List
import jieba
import numpy as np
from utils import Processor, FilePath

class TextProcessor(Processor):
    """文本数据增强处理器"""
    
    def __init__(self):
        super().__init__()
        self.supported_types = {'txt'}
    
    def process(self, input_path: FilePath, output_path: FilePath) -> None:
        """处理单个文件"""
        with open(input_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # 直接复制原文件
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(text)
    
    def augment(self, input_path: FilePath, output_dir: Path) -> List[Path]:
        """对文本进行数据增强，返回增强后的文件路径列表"""
        with open(input_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        augmented_texts = []
        
        # 1. 同义词替换
        augmented_texts.append(self._synonym_replacement(text))
        
        # 2. 随机插入
        augmented_texts.append(self._random_insertion(text))
        
        # 3. 随机交换
        augmented_texts.append(self._random_swap(text))
        
        # 4. 随机删除
        augmented_texts.append(self._random_deletion(text))
        
        # 保存增强后的文本
        output_paths = []
        for aug_text in augmented_texts:
            if aug_text:
                output_path = output_dir / 'description.txt'
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(aug_text)
                output_paths.append(output_path)
        
        return output_paths
    
    def _synonym_replacement(self, text: str, p: float = 0.1) -> str:
        """同义词替换
        
        Args:
            text: 原文本
            p: 替换概率
        """
        words = list(jieba.cut(text))
        n = max(1, int(len(words) * p))
        indices = random.sample(range(len(words)), n)
        
        for i in indices:
            # 这里应该使用同义词词典，这里简单模拟
            words[i] = words[i] + "'"
        
        return ''.join(words)
    
    def _random_insertion(self, text: str, n: int = 3) -> str:
        """随机插入
        
        Args:
            text: 原文本
            n: 插入次数
        """
        words = list(jieba.cut(text))
        for _ in range(n):
            # 随机选择一个位置插入一个随机词
            insert_pos = random.randint(0, len(words))
            insert_word = random.choice(words)
            words.insert(insert_pos, insert_word)
        
        return ''.join(words)
    
    def _random_swap(self, text: str, n: int = 3) -> str:
        """随机交换
        
        Args:
            text: 原文本
            n: 交换次数
        """
        words = list(jieba.cut(text))
        for _ in range(n):
            if len(words) >= 2:
                i, j = random.sample(range(len(words)), 2)
                words[i], words[j] = words[j], words[i]
        
        return ''.join(words)
    
    def _random_deletion(self, text: str, p: float = 0.1) -> str:
        """随机删除
        
        Args:
            text: 原文本
            p: 删除概率
        """
        words = list(jieba.cut(text))
        # 确保至少保留一个词
        if len(words) == 1:
            return text
        
        new_words = []
        for word in words:
            if random.random() > p:
                new_words.append(word)
        
        if not new_words:
            new_words = [random.choice(words)]
        
        return ''.join(new_words) 