from PIL import Image
import numpy as np
import torch

class ImageMetadataExtractor:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("INT", "INT", "FLOAT", "STRING")
    RETURN_NAMES = ("width", "height", "aspect_ratio", "dimensions")
    FUNCTION = "extract"
    CATEGORY = "SmartScaler"

    def extract(self, image):
        # Convert ComfyUI IMAGE to PIL
        img = Image.fromarray((image[0].cpu().numpy() * 255).astype(np.uint8))
        width, height = img.size
        aspect_ratio = width / height if height != 0 else 0.0
        dimensions = f"{width}x{height}"

        return (width, height, aspect_ratio, dimensions)
