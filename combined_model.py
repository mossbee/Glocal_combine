import torch
import torch.nn as nn
from inference import load_pretrained_model, get_embedding
from face_parts_model import MultiFacePartsModel
from PIL import Image
import torchvision.transforms as transforms

class TwinVerificationModel(nn.Module):
    """Combined model for twin verification using global + local features"""
    
    def __init__(self, adaface_arch='ir_50', face_parts_embedding_dim=128, 
                 freeze_adaface=True, final_embedding_dim=512):
        super(TwinVerificationModel, self).__init__()
        
        # Load frozen AdaFace model for global features (512D)
        self.adaface_model = load_pretrained_model(adaface_arch)
        self.freeze_adaface = freeze_adaface
        
        if freeze_adaface:
            # Freeze AdaFace parameters
            for param in self.adaface_model.parameters():
                param.requires_grad = False
            self.adaface_model.eval()
        
        # Face parts model for local features (5 * 128D = 640D)
        self.face_parts_model = MultiFacePartsModel(
            num_parts=5, 
            embedding_dim=face_parts_embedding_dim, 
            pretrained=True
        )
        
        # Dimensions
        self.global_dim = 512  # AdaFace output
        self.local_dim = 5 * face_parts_embedding_dim  # Face parts output
        self.total_dim = self.global_dim + self.local_dim  # Combined: 1152D
        self.final_embedding_dim = final_embedding_dim
        
        # Configure final projection based on desired output dimension
        if final_embedding_dim == self.total_dim:
            # No compression - keep all 1152D features (best for twin verification)
            self.final_projection = nn.Identity()
            print(f"Using full feature dimension: {self.total_dim}D (no compression)")
        elif final_embedding_dim >= 768:
            # Minimal compression for high-quality embeddings
            self.final_projection = nn.Sequential(
                nn.Linear(self.total_dim, final_embedding_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            )
            print(f"Using high-quality embedding: {final_embedding_dim}D")
        elif final_embedding_dim >= 512:
            # Standard compression (recommended for twin verification)
            self.final_projection = nn.Sequential(
                nn.Linear(self.total_dim, 768),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(768, final_embedding_dim)
            )
            print(f"Using standard embedding: {final_embedding_dim}D")
        else:
            # Heavy compression (not recommended for twins, but available)
            self.final_projection = nn.Sequential(
                nn.Linear(self.total_dim, 512),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(512, final_embedding_dim)
            )
            print(f"Warning: Using compressed embedding: {final_embedding_dim}D (may lose important features for twin verification)")
        
    def forward(self, adaface_tensor, face_parts_dict):
        """
        Forward pass combining global and local features
        
        Args:
            adaface_tensor: Preprocessed tensor for AdaFace [batch_size, 3, 112, 112]
            face_parts_dict: Dict of face parts tensors [batch_size, 3, 224, 224]
        
        Returns:
            final_features: [batch_size, final_embedding_dim]
            combined_features: [batch_size, 1152] - full combined features
            global_features: [batch_size, 512] - AdaFace features
            local_features: [batch_size, 640] - Face parts features
        """
        # Extract global features using AdaFace (frozen)
        if self.freeze_adaface:
            with torch.no_grad():
                global_features, _ = self.adaface_model(adaface_tensor)  # [batch_size, 512]
        else:
            global_features, _ = self.adaface_model(adaface_tensor)
        
        # Extract local features using face parts model (trainable)
        local_features = self.face_parts_model(face_parts_dict)  # [batch_size, 640]
        
        # Combine global and local features
        combined_features = torch.cat([global_features, local_features], dim=1)  # [batch_size, 1152]
        
        # Apply final projection to desired embedding dimension
        final_features = self.final_projection(combined_features)
        
        return final_features, combined_features, global_features, local_features

class TripletLoss(nn.Module):
    """Triplet loss for twin verification training"""
    
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.triplet_loss = nn.TripletMarginLoss(margin=margin, p=2.0, reduction='mean')
    
    def forward(self, anchor, positive, negative):
        """
        Compute triplet loss
        
        Args:
            anchor: Features of anchor samples [batch_size, feature_dim]
            positive: Features of positive samples [batch_size, feature_dim]  
            negative: Features of negative samples [batch_size, feature_dim]
        
        Returns:
            loss: Triplet loss value
        """
        return self.triplet_loss(anchor, positive, negative)

class ContrastiveLoss(nn.Module):
    """Contrastive loss for similarity learning"""
    
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
    
    def forward(self, feature1, feature2, label):
        """
        Compute contrastive loss
        
        Args:
            feature1: Features of first sample [batch_size, feature_dim]
            feature2: Features of second sample [batch_size, feature_dim]
            label: 1 if same person, 0 if different [batch_size]
        
        Returns:
            loss: Contrastive loss value
        """
        euclidean_distance = nn.functional.pairwise_distance(feature1, feature2)
        
        # Loss for positive pairs (same person)
        positive_loss = (1 - label) * torch.pow(euclidean_distance, 2)
        
        # Loss for negative pairs (different person)
        negative_loss = label * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        
        loss = torch.mean(positive_loss + negative_loss)
        return loss

def compute_similarity(feature1, feature2, metric='cosine'):
    """
    Compute similarity between two feature vectors
    
    Args:
        feature1: First feature vector [batch_size, feature_dim]
        feature2: Second feature vector [batch_size, feature_dim]
        metric: 'cosine' or 'euclidean'
    
    Returns:
        similarity: Similarity scores [batch_size]
    """
    if metric == 'cosine':
        # Cosine similarity
        similarity = nn.functional.cosine_similarity(feature1, feature2, dim=1)
    elif metric == 'euclidean':
        # Negative euclidean distance (higher = more similar)
        similarity = -nn.functional.pairwise_distance(feature1, feature2)
    else:
        raise ValueError(f"Unknown metric: {metric}")
    
    return similarity

def test_combined_model():
    """Test function for the combined model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    model = TwinVerificationModel(freeze_adaface=True)
    model.to(device)
    
    # Test inputs
    batch_size = 4
    adaface_input = torch.randn(batch_size, 3, 112, 112).to(device)
    face_parts_input = {
        'left_eye': torch.randn(batch_size, 3, 224, 224).to(device),
        'right_eye': torch.randn(batch_size, 3, 224, 224).to(device),
        'nose': torch.randn(batch_size, 3, 224, 224).to(device),
        'mouth': torch.randn(batch_size, 3, 224, 224).to(device),
        'chin': torch.randn(batch_size, 3, 224, 224).to(device)
    }
    
    # Forward pass
    final_features, combined_features, global_features, local_features = model(
        adaface_input, face_parts_input
    )
    
    print(f"Global features shape: {global_features.shape}")  # [4, 512]
    print(f"Local features shape: {local_features.shape}")    # [4, 640]
    print(f"Combined features shape: {combined_features.shape}")  # [4, 1152]
    print(f"Final features shape: {final_features.shape}")    # [4, 512]
    
    # Test triplet loss
    triplet_loss = TripletLoss(margin=1.0)
    anchor = final_features[:2]
    positive = final_features[1:3]
    negative = final_features[2:4]
    
    loss = triplet_loss(anchor, positive, negative)
    print(f"Triplet loss: {loss.item()}")

if __name__ == "__main__":
    test_combined_model() 