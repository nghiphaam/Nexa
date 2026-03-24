"""Image preprocessing for SigLIP."""
from PIL import Image
import torch


class ImageProcessor:
    def __init__(self, model_name: str = "google/siglip-base-patch16-224"):
        from transformers import AutoProcessor
        self.processor = AutoProcessor.from_pretrained(model_name)

    def __call__(self, image):
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, Image.Image):
            image = image.convert("RGB")

        inputs = self.processor(images=image, return_tensors="pt")
        return inputs["pixel_values"]

    def batch_process(self, images):
        pil_images = []
        for img in images:
            if isinstance(img, str):
                pil_images.append(Image.open(img).convert("RGB"))
            elif isinstance(img, Image.Image):
                pil_images.append(img.convert("RGB"))
            else:
                pil_images.append(img)

        inputs = self.processor(images=pil_images, return_tensors="pt")
        return inputs["pixel_values"]
