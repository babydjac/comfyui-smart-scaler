import torch
from PIL import Image
import numpy as np

class BatchFrameProcessor:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),  # Batch of images (B, H, W, C)
                "target_resolution": (["SDXL", "Wan-Small", "Wan-Medium", "Wan-Large"], {"default": "Wan-Small"}),
                "fit_strategy": (["resize", "pad", "crop"], {"default": "resize"}),
            },
            "optional": {
                "force_multiple_64": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("scaled_images", "scaled_size")
    FUNCTION = "process"
    CATEGORY = "SmartScaler/Video"

    def process(self, images, target_resolution, fit_strategy, force_multiple_64=True):
        # Define target resolutions
        resolution_map = {
            "SDXL": [(1024, 1536), (1536, 1024), (1024, 1024)],
            "Wan-Small": [(512, 768), (768, 512), (512, 512)],
            "Wan-Medium": [(768, 1152), (1152, 768), (768, 768)],
            "Wan-Large": [(1024, 1536), (1536, 1024), (1024, 1024)]
        }
        targets = resolution_map[target_resolution]

        # Process the first image to determine target size
        first_img = Image.fromarray((images[0].cpu().numpy() * 255).astype(np.uint8))
        orig_width, orig_height = first_img.size
        aspect_ratio = orig_width / orig_height

        # Select closest target resolution
        def get_closest_target(aspect, targets):
            def aspect_diff(target): return abs(aspect - target[0] / target[1])
            return min(targets, key=aspect_diff)

        target_size = get_closest_target(aspect_ratio, targets)
        target_w, target_h = target_size

        # Process all images
        scaled_images = []
        for img_tensor in images:
            img = Image.fromarray((img_tensor.cpu().numpy() * 255).astype(np.uint8))

            # Calculate scaling to fill target while preserving aspect ratio
            scale = max(target_w / orig_width, target_h / orig_height)
            new_w, new_h = int(orig_width * scale), int(orig_height * scale)

            # Ensure dimensions are multiples of 64 if required
            if force_multiple_64:
                new_w = (new_w // 64) * 64
                new_h = (new_h // 64) * 64
                scale = max(new_w / orig_width, new_h / orig_height)
                new_w, new_h = int(orig_width * scale), int(orig_height * scale)

            # Resize
            img_resized = img.resize((new_w, new_h), Image.LANCZOS)

            # Apply fit strategy
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

            # Convert back to tensor
            np_img = np.array(img_resized).astype(np.float32) / 255.0
            scaled_tensor = torch.from_numpy(np_img)
            scaled_images.append(scaled_tensor)

        # Stack images into a batch
        scaled_batch = torch.stack(scaled_images)

        # Prepare size string
        scaled_size = f"{target_w}x{target_h}"

        return (scaled_batch, scaled_size)
