import torch
import torchvision.transforms as transforms
from torchvision.models import models
from PIL import Image

model = models.resnet50(pretrained=models.ResNet50_Weights.IMAGENET1K_V2)
model.eval()
embedding_model = torch.nn.Sequential(*list(model.children())[:-1])  # Remove the last layer
# This will give us a feature vector of size [1, 2048] for the input image
# Add a linear projection layer to reduce output to 128 dimensions
embedding_model = torch.nn.Sequential(
    embedding_model,
    torch.nn.Flatten(),
    torch.nn.Linear(2048, 128)  # Project to 128 dimensions
)

# Define the preprocessing steps

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

image_path = '/your/image/path.jpg'  # Replace with your image path
image = Image.open(image_path).convert('RGB')
input_tensor = preprocess(image).unsqueeze(0)  # Shape [1, 128]