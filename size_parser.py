class SizeParser:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sdxl_size": ("STRING", {"forceInput": True}),
                "wan_size": ("STRING", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("INT", "INT", "INT", "INT")
    RETURN_NAMES = ("sdxl_width", "sdxl_height", "wan_width", "wan_height")
    FUNCTION = "parse"
    CATEGORY = "SmartScaler"

    def parse(self, sdxl_size, wan_size):
        # Initialize defaults
        sdxl_width, sdxl_height = 0, 0
        wan_width, wan_height = 0, 0

        # Parse SDXL size
        if sdxl_size != "N/A":
            try:
                w, h = map(int, sdxl_size.split("x"))
                sdxl_width, sdxl_height = w, h
            except (ValueError, AttributeError):
                pass  # Keep defaults if parsing fails

        # Parse Wan size
        if wan_size != "N/A":
            try:
                w, h = map(int, wan_size.split("x"))
                wan_width, wan_height = w, h
            except (ValueError, AttributeError):
                pass  # Keep defaults if parsing fails

        return (sdxl_width, sdxl_height, wan_width, wan_height)