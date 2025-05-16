import torch
from PIL import Image
import numpy as np

class SmartAspectScaler:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "target_models": (["SDXL", "Wan", "Both"], {"default": "Both"}),
                "wan_size": (["Small", "Medium", "Large"], {"default": "Small"}),
                "fit_strategy": (["resize", "pad", "crop"], {"default": "resize"}),
            },
            "optional": {
                "force_multiple_64": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "INT", "INT", "STRING", "STRING")
    RETURN_NAMES = ("sdxl_image", "wan_image", "original_width", "original_height", "scaled_sdxl_size", "scaled_wan_size")
    FUNCTION = "scale"
    CATEGORY = "SmartScaler"

    def scale(self, image, target_models, wan_size, fit_strategy, force_multiple_64=True):
        # Convert ComfyUI IMAGE (batch, H, W, C) to PIL
        img = Image.fromarray((image[0].cpu().numpy() * 255).astype(np.uint8))
        orig_width, orig_height = img.size
        aspect_ratio = orig_width / orig_height

        # Define target resolutions
        sdxl_targets = [(1024, 1536), (1536, 1024), (1024, 1024)]  # Portrait, Landscape, Square

        # Wan sizes: Small (512x768), Medium (768x1152), Large (1024x1536)
        wan_size_map = {
            "Small": [(512, 768), (768, 512), (512, 512)],   # Portrait, Landscape, Square
            "Medium": [(768, 1152), (1152, 768), (768, 768)],
            "Large": [(1024, 1536), (1536, 1024), (1024, 1024)]
        }
        wan_targets = wan_size_map[wan_size]

        # Select closest target resolution based on aspect ratio
        def get_closest_target(aspect, targets):
            def aspect_diff(target): return abs(aspect - target[0] / target[1])
            return min(targets, key=aspect_diff)

        sdxl_size = get_closest_target(aspect_ratio, sdxl_targets) if target_models in ["SDXL", "Both"] else None
        wan_size = get_closest_target(aspect_ratio, wan_targets) if target_models in ["Wan", "Both"] else None

        def resize_image(target_size, strategy):
            if not target_size:
                return None
            target_w, target_h = target_size

            # Calculate scaling to fill target while preserving aspect ratio
            scale = max(target_w / orig_width, target_h / orig_height)  # Fill, not fit
            new_w, new_h = int(orig_width * scale), int(orig_height * scale)

            # Ensure dimensions are multiples of 64 if required
            if force_multiple_64:
                new_w = (new_w // 64) * 64
                new_h = (new_h // 64) * 64
                # Recalculate scale to maintain aspect ratio
                scale = max(new_w / orig_width, new_h / orig_height)
                new_w, new_h = int(orig_width * scale), int(orig_height * scale)

            # Resize to calculated dimensions
            img_resized = img.resize((new_w, new_h), Image.LANCZOS)

            # Apply fit strategy to match target dimensions
            if new_w != target_w or new_h != target_h:
                if strategy == "pad":
                    new_img = Image.new("RGB", (target_w, target_h), (0, 0, 0))
                    offset = ((target_w - new_w) // 2, (target_h - new_h) // 2)
                    new_img.paste(img_resized, offset)
                    img_resized = new_img
                elif strategy == "crop":
                    left = (new_w - target_w) // 2 if new_w > target_w else 0
                    top = (new_h - target_h) // 2 if new_h > target_h else 0
                    img_resized = img_resized.crop((left, top, left + target_w, top + target_h))
                # "resize" strategy: no further stretch, keep scaled size

            return img_resized

        # Resize for SDXL and Wan
        sdxl_img = resize_image(sdxl_size, fit_strategy)
        wan_img = resize_image(wan_size, fit_strategy)

        # Convert back to ComfyUI IMAGE
        def to_comfy_image(pil_img):
            if pil_img is None:
                return torch.zeros((1, 512, 512, 3), dtype=torch.float32)
            np_img = np.array(pil_img).astype(np.float32) / 255.0
            return torch.from_numpy(np_img).unsqueeze(0)

        sdxl_tensor = to_comfy_image(sdxl_img)
        wan_tensor = to_comfy_image(wan_img)

        # Prepare size strings
        sdxl_size_str = f"{sdxl_img.size[0]}x{sdxl_img.size[1]}" if sdxl_img else "N/A"
        wan_size_str = f"{wan_img.size[0]}x{wan_img.size[1]}" if wan_img else "N/A"

        return (sdxl_tensor, wan_tensor, orig_width, orig_height, sdxl_size_str, wan_size_str)