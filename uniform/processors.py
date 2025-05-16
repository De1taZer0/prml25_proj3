from PIL import Image
import PyPDF2
import textract
from tqdm import tqdm
from utils.file import FilePath, FileCategory, detect_encoding
from utils.word import get_word_app, word_lock
from typing import List, Type

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

class ImageProcessor(Processor):
    """图片处理器"""
    def __init__(self):
        super().__init__()
        self.supported_types = {}
        self.supported_categories = {FileCategory.IMAGE}

    def process(self, input_path: FilePath, output_path: FilePath) -> None:
        """处理图片文件"""
        with Image.open(input_path) as img:
            if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
                img = img.convert('RGB')
            
            target_width = 960
            target_height = 540
            width, height = img.size
            
            ratio = min(target_width/width, target_height/height)
            new_size = (int(width * ratio), int(height * ratio))
            
            img = img.resize(new_size, Image.Resampling.LANCZOS)
            
            new_img = Image.new('RGB', (target_width, target_height), 'white')
            
            paste_x = (target_width - new_size[0]) // 2
            paste_y = (target_height - new_size[1]) // 2
            new_img.paste(img, (paste_x, paste_y))
            
            new_img.save(output_path, "PNG")

class PDFProcessor(Processor):
    """PDF处理器"""
    def __init__(self):
        super().__init__()
        self.supported_types = {'pdf'}
        self.supported_categories = {}

    def process(self, input_path: FilePath, output_path: FilePath) -> None:
        """处理PDF文件"""
        pdf = PyPDF2.PdfReader(input_path)
        text = "".join(page.extract_text() for page in pdf.pages)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(text)

class DOCXProcessor(Processor):
    """DOCX处理器"""
    def __init__(self):
        super().__init__()
        self.supported_types = {'docx'}
        self.supported_categories = {}

    def process(self, input_path: FilePath, output_path: FilePath) -> None:
        """处理DOCX文件"""
        input_path_str = str(input_path.absolute())
        text = textract.process(input_path_str).decode('utf-8')
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(text)

class DOCProcessor(Processor):
    """DOC处理器"""
    def __init__(self):
        super().__init__()
        self.supported_types = {'doc'}
        self.supported_categories = {}

    def process(self, input_path: FilePath, output_path: FilePath) -> None:
        """处理DOC文件"""
        with word_lock:
            word = get_word_app()
            doc = word.Documents.Open(str(input_path.absolute()))
            text = doc.Content.Text
            doc.Close()
            if text:
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(text)

class TxtProcessor(Processor):
    """文本处理器"""
    def __init__(self):
        super().__init__()
        self.supported_types = {'txt'}
        self.supported_categories = {}

    def process(self, input_path: FilePath, output_path: FilePath) -> None:
        """处理文本文件"""
        try:
            encoding = detect_encoding(input_path)
            with open(input_path, "r", encoding=encoding) as f:
                text = f.read()
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(text)
        except UnicodeDecodeError:
            tqdm.write(f"无法解码文件: {input_path}，检测到的编码: {encoding}")
        except Exception as e:
            tqdm.write(f"处理文本文件 {input_path} 时发生错误: {e}")


# 处理器类型列表
PROCESSOR_TYPES: List[Type[Processor]] = [
    ImageProcessor,
    PDFProcessor,
    DOCXProcessor,
    DOCProcessor,
    TxtProcessor
]

def create_processor(file_path: FilePath) -> Processor:
    """创建适合的处理器实例"""
    for processor_type in PROCESSOR_TYPES:
        processor = processor_type()
        if processor.can_process(file_path):
            return processor
    raise ValueError(f"没有找到适合的处理器来处理文件: {file_path}") 