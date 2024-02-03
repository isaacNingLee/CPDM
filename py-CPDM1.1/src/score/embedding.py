import torch
from torchvision import transforms
from torchvision.models import clip
from PIL import Image

class CLIPPytorch:
    def __init__(self, model_name='ViT-L/14@336px'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.model.eval()

    def _preprocess_images(self, images):
        """
        Preprocess images using the CLIP specific transforms.
        """
        return [self.preprocess(Image.fromarray((img * 255).astype('uint8'))).unsqueeze(0) for img in images]

    def embed(self, images):
        """
        Computes CLIP embeddings for the given images.

        Args:
          images: A list of PIL Images.

        Returns:
          Embedding tensor of shape (batch_size, embedding_width).
        """
        images = torch.cat(self._preprocess_images(images)).to(self.device)
        with torch.no_grad():
            image_features = self.model.encode_image(images)
        return image_features
