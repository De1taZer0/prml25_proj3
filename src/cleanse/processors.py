from PIL import Image
import re
from utils.file import FilePath, FileCategory
from utils.processor import Processor
from typing import List, Type


class ImageProcessor(Processor):
    """图片处理器"""
    def __init__(self):
        super().__init__()
        self.supported_types = {'png'}
        self.supported_categories = {}

    def process(self, input_path: FilePath, output_path: FilePath) -> None:
        """清洗图片文件"""
        with Image.open(input_path) as img:
            img.save(output_path, "PNG")

class TextProcessor(Processor):
    """文本处理器"""
    def __init__(self):
        super().__init__()
        self.supported_types = {'txt'}
        self.supported_categories = {}

    def process(self, input_path: FilePath, output_path: FilePath) -> None:
        """清洗文本文件"""
        with open(input_path, "r", encoding="utf-8") as f:
            text = f.read()
        text = re.sub(r"[^\u4e00-\u9fa5a-zA-Z]", "", text)
        text = text.strip()
        text = re.sub(r"\n\s*\n", "\n", text)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(text)

# 处理器类型列表
PROCESSOR_TYPES: List[Type[Processor]] = [
    ImageProcessor,
    TextProcessor
]

def create_processor(file_path: FilePath) -> Processor:
    """创建适合的处理器实例"""
    for processor_type in PROCESSOR_TYPES:
        processor = processor_type()
        if processor.can_process(file_path):
            return processor
    raise ValueError(f"没有找到适合的处理器来处理文件: {file_path}") 