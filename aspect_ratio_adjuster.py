from PIL import Image
import numpy as np
import torch

class AspectRatioAdjuster:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "target_ratio": (["16:9", "4:3", "1:1", "Custom"], {"default": "16:9"}),
                "custom_ratio_width": ("FLOAT", {"default": 16.0, "min": 1.0, "max": 100.0, "step": 0.1}),
                "custom_ratio_height": ("FLOAT", {"default": 9.0, "min": 1.0, "max": 100.0, "step": 0.1}),
                "method": (["crop", "pad"], {"default": "crop"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("adjusted_image", "new_ratio")
    FUNCTION = "adjust"
    CATEGORY = "SmartScaler"

    def adjust(self, image, target_ratio, custom_ratio_width, custom_ratio_height, method):
        # Convert ComfyUI IMAGE to PIL
        img = Image.fromarray((image[0].cpu().numpy() * 255).astype(np.uint8))
        orig_width, orig_height = img.size
        orig_ratio = orig_width / orig_height

        # Determine target ratio
        if target_ratio == "Custom":
            target_ratio_value = custom_ratio_width / custom_ratio_height
        else:
            ratio_map = {"16:9": 16/9, "4:3": 4/3, "1:1": 1}
            target_ratio_value = ratio_map[target_ratio]

        # Calculate new dimensions
        if method == "crop":
            if orig_ratio > target_ratio_value:
                # Crop width
                new_width = int(orig_height * target_ratio_value)
                left = (orig_width - new_width) // 2
                img = img.crop((left, 0, left + new_width, orig_height))
            else:
                # Crop height
                new_height = int(orig_width / target_ratio_value)
                top = (orig_height - new_height) // 2
                img = img.crop((0, top, orig_width, top + new_height))
        else:  # pad
            if orig_ratio > target_ratio_value:
                # Pad height
                new_height = int(orig_width / target_ratio_value)
                new_img = Image.new("RGB", (orig_width, new_height), (0, 0, 0))
                new_img.paste(img, (0, (new_height - orig_height) // 2))
                img = new_img
            else:
                # Pad width
                new_width = int(orig_height * target_ratio_value)
                new_img = Image.new("RGB", (new_width, orig_height), (0, 0, 0))
                new_img.paste(img, ((new_width - orig_width) // 2, 0))
                img = new_img

        # Convert back to ComfyUI IMAGE
        np_img = np.array(img).astype(np.float32) / 255.0
        adjusted_tensor = torch.from_numpy(np_img).unsqueeze(0)

        # Return new ratio as string
        new_ratio = f"{img.size[0]}:{img.size[1]}"
        return (adjusted_tensor, new_ratio)
