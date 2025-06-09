from pathlib import Path
from enum import Enum, auto
from typing import Iterator
import chardet

class FileCategory(Enum):
    IMAGE = auto()
    TEXT = auto()
    OTHER = auto()

    @classmethod
    def from_extension(cls, ext: str) -> 'FileCategory':
        if ext in {"png", "jpg", "jpeg"}:
            return cls.IMAGE
        if ext in {"txt", "doc", "docx", "pdf", "rtf"}:
            return cls.TEXT
        return cls.OTHER

class FilePath(Path):
    def __new__(cls, *args, **kwargs):
        return super().__new__(cls, *args, **kwargs)
    
    @property
    def extension(self) -> str:
        return self.suffix[1:].lower()
    
    @property
    def category(self) -> FileCategory:
        return FileCategory.from_extension(self.extension)
    
    @property
    def is_image(self) -> bool:
        return self.category == FileCategory.IMAGE
    
    @property
    def is_text(self) -> bool:
        return self.category == FileCategory.TEXT

    @property
    def valid(self) -> bool:
        return self.is_image or self.is_text

def get_all_files(dir_path: Path, full_path: bool = False) -> Iterator[FilePath]:
    for path in dir_path.rglob("*"):
        if path.is_file():
            yield FilePath(str(path) if full_path else path.name)

def get_output_path(input_path: FilePath, data_processed_dir: Path) -> FilePath:
    """生成输出文件路径，只保留最上层和最下层目录"""
    rel_path = input_path.relative_to(data_processed_dir.parent).parts[1]
    output_dir = data_processed_dir / rel_path
    
    if input_path.is_image:
        output_name = "formula.png"
    elif input_path.is_text:
        output_name = "description.txt"
    else:
        assert False, f"未知文件类型: {input_path}"
    
    return FilePath(output_dir / output_name) 

def detect_encoding(file_path: Path) -> str:
    """检测文件编码"""
    with open(file_path, 'rb') as f:
        raw_data = f.read()
        result = chardet.detect(raw_data)
        return result['encoding'] or 'utf-8'
