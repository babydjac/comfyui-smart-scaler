class DynamicResolutionSelector:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sdxl_width": ("INT", {"default": 1024, "min": 64, "max": 4096, "step": 64}),
                "sdxl_height": ("INT", {"default": 1536, "min": 64, "max": 4096, "step": 64}),
                "wan_width": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 64}),
                "wan_height": ("INT", {"default": 768, "min": 64, "max": 2048, "step": 64}),
                "force_multiple_64": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("sdxl_resolution", "wan_resolution")
    FUNCTION = "select"
    CATEGORY = "SmartScaler"

    def select(self, sdxl_width, sdxl_height, wan_width, wan_height, force_multiple_64):
        # Ensure multiples of 64 if required
        if force_multiple_64:
            sdxl_width = (sdxl_width // 64) * 64
            sdxl_height = (sdxl_height // 64) * 64
            wan_width = (wan_width // 64) * 64
            wan_height = (wan_height // 64) * 64

        # Ensure non-zero dimensions
        sdxl_width = max(64, sdxl_width)
        sdxl_height = max(64, sdxl_height)
        wan_width = max(64, wan_width)
        wan_height = max(64, wan_height)

        sdxl_resolution = f"{sdxl_width}x{sdxl_height}"
        wan_resolution = f"{wan_width}x{wan_height}"

        return (sdxl_resolution, wan_resolution)
