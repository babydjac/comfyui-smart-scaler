from .smart_aspect_scaler import SmartAspectScaler
from .size_parser import SizeParser
from .aspect_ratio_adjuster import AspectRatioAdjuster
from .image_metadata_extractor import ImageMetadataExtractor
from .dynamic_resolution_selector import DynamicResolutionSelector
from .wan_video_frame_scaler import WanVideoFrameScaler
from .batch_frame_processor import BatchFrameProcessor

NODE_CLASS_MAPPINGS = {
    "SmartAspectScaler": SmartAspectScaler,
    "SizeParser": SizeParser,
    "AspectRatioAdjuster": AspectRatioAdjuster,
    "ImageMetadataExtractor": ImageMetadataExtractor,
    "DynamicResolutionSelector": DynamicResolutionSelector,
    "WanVideoFrameScaler": WanVideoFrameScaler,
    "BatchFrameProcessor": BatchFrameProcessor
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SmartAspectScaler": "Smart Aspect Scaler",
    "SizeParser": "Size Parser",
    "AspectRatioAdjuster": "Aspect Ratio Adjuster",
    "ImageMetadataExtractor": "Image Metadata Extractor",
    "DynamicResolutionSelector": "Dynamic Resolution Selector",
    "WanVideoFrameScaler": "Wan Video Frame Scaler",
    "BatchFrameProcessor": "Batch Frame Processor"
}

print("\033[34mComfyUI Smart Scaler Package: \033[92mLoaded\033[0m")