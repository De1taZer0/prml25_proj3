"""
数据增强处理器
"""
from typing import Optional
from utils import FilePath, FileCategory, Processor
from .text_processor import TextProcessor
from .image_processor import ImageProcessor

def create_processor(file_path: FilePath) -> Optional[Processor]:
    """根据文件类型创建相应的处理器"""
    if not isinstance(file_path, FilePath):
        file_path = FilePath(file_path)
    
    category = FileCategory.from_extension(file_path.extension)
    
    if category == FileCategory.TEXT:
        return TextProcessor()
    elif category == FileCategory.IMAGE:
        return ImageProcessor()
    else:
        raise ValueError(f"不支持的文件类型: {file_path}")

__all__ = ['ImageProcessor', 'TextProcessor']

