from pathlib import Path
from typing import List
import numpy as np
from PIL import Image, ImageEnhance
from utils import Processor, FilePath

class ImageProcessor(Processor):
    """图像数据增强处理器"""
    
    def __init__(self):
        super().__init__()
        self.supported_types = {'png'}
    
    def process(self, input_path: FilePath, output_path: FilePath) -> None:
        """处理单个文件"""
        # 直接复制原图像
        image = Image.open(input_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path)
    
    def augment(self, input_path: FilePath, output_dir: Path) -> List[Path]:
        """对图像进行数据增强，返回增强后的文件路径列表"""
        image = Image.open(input_path)
        output_paths = []
        
        # 1. 旋转
        for angle in [90, 180, 270]:
            rotated = self._rotate_image(image, angle)
            output_path = output_dir / f"formula_rotate_{angle}.png"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            rotated.save(output_path)
            output_paths.append(output_path)
        
        # 2. 翻转
        flipped_h = self._flip_image(image, True)
        flipped_v = self._flip_image(image, False)
        
        output_path_h = output_dir / "formula_flip_h.png"
        output_path_v = output_dir / "formula_flip_v.png"
        output_path_h.parent.mkdir(parents=True, exist_ok=True)
        output_path_v.parent.mkdir(parents=True, exist_ok=True)
        
        flipped_h.save(output_path_h)
        flipped_v.save(output_path_v)
        output_paths.extend([output_path_h, output_path_v])
        
        # 3. 亮度调整
        for factor in [0.8, 1.2]:
            brightened = self._adjust_brightness(image, factor)
            output_path = output_dir / f"formula_brightness_{factor}.png"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            brightened.save(output_path)
            output_paths.append(output_path)
        
        # 4. 对比度调整
        for factor in [0.8, 1.2]:
            contrasted = self._adjust_contrast(image, factor)
            output_path = output_dir / f"formula_contrast_{factor}.png"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            contrasted.save(output_path)
            output_paths.append(output_path)
        
        # 5. 添加噪声
        noisy = self._add_noise(image)
        output_path = output_dir / "formula_noise.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        noisy.save(output_path)
        output_paths.append(output_path)
        
        # 选择最后一个生成的图像作为formula.png
        if output_paths:
            final_output = output_dir / "formula.png"
            final_output.parent.mkdir(parents=True, exist_ok=True)
            Image.open(output_paths[-1]).save(final_output)
            output_paths.append(final_output)
        
        return output_paths
    
    def _rotate_image(self, image: Image.Image, angle: float) -> Image.Image:
        """旋转图像
        
        Args:
            image: 原图像
            angle: 旋转角度
        """
        return image.rotate(angle, expand=True)
    
    def _flip_image(self, image: Image.Image, horizontal: bool) -> Image.Image:
        """翻转图像
        
        Args:
            image: 原图像
            horizontal: 是否水平翻转，False为垂直翻转
        """
        if horizontal:
            return image.transpose(Image.FLIP_LEFT_RIGHT)
        else:
            return image.transpose(Image.FLIP_TOP_BOTTOM)
    
    def _adjust_brightness(self, image: Image.Image, factor: float) -> Image.Image:
        """调整亮度
        
        Args:
            image: 原图像
            factor: 亮度因子
        """
        enhancer = ImageEnhance.Brightness(image)
        return enhancer.enhance(factor)
    
    def _adjust_contrast(self, image: Image.Image, factor: float) -> Image.Image:
        """调整对比度
        
        Args:
            image: 原图像
            factor: 对比度因子
        """
        enhancer = ImageEnhance.Contrast(image)
        return enhancer.enhance(factor)
    
    def _add_noise(self, image: Image.Image, intensity: float = 0.1) -> Image.Image:
        """添加噪声
        
        Args:
            image: 原图像
            intensity: 噪声强度
        """
        img_array = np.array(image)
        noise = np.random.normal(0, intensity * 255, img_array.shape)
        noisy_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy_array) 