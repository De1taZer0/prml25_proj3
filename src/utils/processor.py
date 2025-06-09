from .file import FilePath

class Processor:
    """处理器基类"""
    def __init__(self):
        self.supported_types = set()
        self.supported_categories = set()

    def can_process(self, file_path: FilePath) -> bool:
        """检查是否可以处理该文件"""
        return (file_path.extension in self.supported_types or 
                file_path.category in self.supported_categories)

    def process(self, input_path: FilePath, output_path: FilePath) -> None:
        """处理文件"""
        raise NotImplementedError
