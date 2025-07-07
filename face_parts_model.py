import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image

class FacePartsResNet50(nn.Module):
    """Enhanced ResNet50 model for face parts feature extraction"""
    
    def __init__(self, embedding_dim=128, pretrained=True):
        super(FacePartsResNet50, self).__init__()
        
        # Load pretrained ResNet50
        if pretrained:
            self.backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        else:
            self.backbone = resnet50(weights=None)
        
        # Remove the final classification layer and avgpool
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])
        
        # Add adaptive pooling and projection layers
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.projection = nn.Linear(2048, embedding_dim)
        self.dropout = nn.Dropout(0.1)
        
        # Initialize projection layer
        nn.init.kaiming_normal_(self.projection.weight, mode='fan_out', nonlinearity='relu')
        if self.projection.bias is not None:
            nn.init.constant_(self.projection.bias, 0)
    
    def forward(self, x):
        """
        Forward pass for face parts
        Args:
            x: Input tensor [batch_size, 3, 224, 224]
        Returns:
            features: [batch_size, embedding_dim]
        """
        # Extract features using ResNet backbone
        features = self.backbone(x)  # [batch_size, 2048, 7, 7]
        
        # Global average pooling
        features = self.global_pool(features)  # [batch_size, 2048, 1, 1]
        features = self.flatten(features)  # [batch_size, 2048]
        
        # Apply dropout and projection
        features = self.dropout(features)
        features = self.projection(features)  # [batch_size, embedding_dim]
        
        return features

class MultiFacePartsModel(nn.Module):
    """Model for processing multiple face parts simultaneously"""
    
    def __init__(self, num_parts=5, embedding_dim=128, pretrained=True):
        super(MultiFacePartsModel, self).__init__()
        
        self.num_parts = num_parts
        self.embedding_dim = embedding_dim
        
        # Create separate ResNet50 for each face part
        self.part_models = nn.ModuleDict({
            'left_eye': FacePartsResNet50(embedding_dim, pretrained),
            'right_eye': FacePartsResNet50(embedding_dim, pretrained),
            'nose': FacePartsResNet50(embedding_dim, pretrained),
            'mouth': FacePartsResNet50(embedding_dim, pretrained),
            'chin': FacePartsResNet50(embedding_dim, pretrained)
        })
        
        self.part_names = ['left_eye', 'right_eye', 'nose', 'mouth', 'chin']
    
    def forward(self, face_parts_dict):
        """
        Forward pass for all face parts
        Args:
            face_parts_dict: Dict with keys ['left_eye', 'right_eye', 'nose', 'mouth', 'chin']
                           Each value is tensor [batch_size, 3, 224, 224]
        Returns:
            features: [batch_size, num_parts * embedding_dim]
        """
        part_features = []
        
        for part_name in self.part_names:
            if part_name in face_parts_dict:
                features = self.part_models[part_name](face_parts_dict[part_name])
                part_features.append(features)
            else:
                # If part is missing, use zeros
                batch_size = list(face_parts_dict.values())[0].size(0)
                zeros = torch.zeros(batch_size, self.embedding_dim, 
                                  device=list(face_parts_dict.values())[0].device)
                part_features.append(zeros)
        
        # Concatenate all part features
        combined_features = torch.cat(part_features, dim=1)  # [batch_size, num_parts * embedding_dim]
        
        return combined_features

def get_face_parts_transforms():
    """Get preprocessing transforms for face parts"""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

def test_models():
    """Test function for the face parts models"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test single face part model
    single_model = FacePartsResNet50(embedding_dim=128)
    single_model.to(device)
    
    # Test input
    test_input = torch.randn(4, 3, 224, 224).to(device)
    output = single_model(test_input)
    print(f"Single part model output shape: {output.shape}")
    
    # Test multi-face parts model
    multi_model = MultiFacePartsModel(num_parts=5, embedding_dim=128)
    multi_model.to(device)
    
    # Test input dictionary
    face_parts_input = {
        'left_eye': torch.randn(4, 3, 224, 224).to(device),
        'right_eye': torch.randn(4, 3, 224, 224).to(device),
        'nose': torch.randn(4, 3, 224, 224).to(device),
        'mouth': torch.randn(4, 3, 224, 224).to(device),
        'chin': torch.randn(4, 3, 224, 224).to(device)
    }
    
    output = multi_model(face_parts_input)
    print(f"Multi-part model output shape: {output.shape}")  # Should be [4, 640]

if __name__ == "__main__":
    test_models() 