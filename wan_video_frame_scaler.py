import torch
from PIL import Image
import numpy as np

class WanVideoFrameScaler:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "wan_resolution": (["720p", "480p", "Custom"], {"default": "720p"}),
                "custom_width": ("INT", {"default": 1280, "min": 64, "max": 4096, "step": 64}),
                "custom_height": ("INT", {"default": 720, "min": 64, "max": 4096, "step": 64}),
                "fit_strategy": (["resize", "pad", "crop"], {"default": "resize"}),
            },
            "optional": {
                "force_multiple_64": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("scaled_image", "scaled_size")
    FUNCTION = "scale"
    CATEGORY = "SmartScaler/Video"

    def scale(self, image, wan_resolution, custom_width, custom_height, fit_strategy, force_multiple_64=True):
        # Convert ComfyUI IMAGE to PIL
        img = Image.fromarray((image[0].cpu().numpy() * 255).astype(np.uint8))
        orig_width, orig_height = img.size
        aspect_ratio = orig_width / orig_height

        # Define Wan 2.1 resolutions
        resolution_map = {
            "720p": (1280, 720),
            "480p": (854, 480)
        }
        target_size = resolution_map.get(wan_resolution, (custom_width, custom_height))

        # Select closest target resolution based on aspect ratio
        target_w, target_h = target_size
        target_ratio = target_w / target_h

        # Calculate scaling to fill target while preserving aspect ratio
        scale = max(target_w / orig_width, target_h / orig_height)  # Fill, not fit
        new_w, new_h = int(orig_width * scale), int(orig_height * scale)

        # Ensure dimensions are multiples of 64 if required
        if force_multiple_64:
            new_w = (new_w // 64) * 64
            new_h = (new_h // 64) * 64
            scale = max(new_w / orig_width, new_h / orig_height)
            new_w, new_h = int(orig_width * scale), int(orig_height * scale)

        # Resize to calculated dimensions
        img_resized = img.resize((new_w, new_h), Image.LANCZOS)

        # Apply fit strategy to match target dimensions
        if new_w != target_w or new_h != target_h:
            if fit_strategy == "pad":
                new_img = Image.new("RGB", (target_w, target_h), (0, 0, 0))
                offset = ((target_w - new_w) // 2, (target_h - new_h) // 2)
                new_img.paste(img_resized, offset)
                img_resized = new_img
            elif fit_strategy == "crop":
                left = (new_w - target_w) // 2 if new_w > target_w else 0
                top = (new_h - target_h) // 2 if new_h > target_h else 0
                img_resized = img_resized.crop((left, top, left + target_w, top + target_h))

        # Convert back to ComfyUI IMAGE
        np_img = np.array(img_resized).astype(np.float32) / 255.0
        scaled_tensor = torch.from_numpy(np_img).unsqueeze(0)

        # Prepare size string
        scaled_size = f"{img_resized.size[0]}x{img_resized.size[1]}"

        return (scaled_tensor, scaled_size)
