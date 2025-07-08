import os
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

from combined_model import TwinVerificationModel
from cutout import get_position, square_from_side_midpoints, warp_perspective_cutout
from face_parts_model import get_face_parts_transforms

class TwinInference:
    """Inference class for the trained twin verification model"""
    
    def __init__(self, model_path, device=None):
        """
        Initialize the inference model
        
        Args:
            model_path: Path to the trained model checkpoint
            device: Device to run inference on
        """
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load checkpoint to get model configuration
        checkpoint = torch.load(model_path, map_location=self.device)
        config = checkpoint.get('config', {})
        
        # Initialize model with saved configuration
        self.model = TwinVerificationModel(
            adaface_arch=config.get('adaface_arch', 'ir_50'),
            face_parts_embedding_dim=config.get('face_parts_embedding_dim', 128),
            freeze_adaface=config.get('freeze_adaface', True),
            final_embedding_dim=config.get('final_embedding_dim', 512)
        )
        
        # Load model weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # Store embedding dimension for reference
        self.embedding_dim = config.get('final_embedding_dim', 512)
        
        # Face parts configuration
        self.face_parts = {
            "left_eye": [35, 168],
            "right_eye": [168, 265], 
            "mouth": [61, 291],
            "nose": [36, 266],
            "chin": [32, 262]
        }
        
        # Set up transforms
        self.face_parts_transform = get_face_parts_transforms()
        
        print(f"Twin verification model loaded on {self.device}")
        print(f"Embedding dimension: {self.embedding_dim}D")
        print(f"Model configuration: {config.get('adaface_arch', 'ir_50')} + face parts")
    
    def _load_tensor_file(self, tensor_path):
        """Load .pt tensor file for AdaFace"""
        try:
            tensor = torch.load(tensor_path, map_location=self.device)
            
            # Handle different tensor shapes
            if tensor.dim() == 4:  # [1, 3, 112, 112]
                tensor = tensor.squeeze(0)  # Remove batch dimension -> [3, 112, 112]
            elif tensor.dim() == 3:  # [3, 112, 112] - already correct
                pass
            elif tensor.dim() == 2:  # Flattened case, need to reshape
                if tensor.numel() == 3 * 112 * 112:
                    tensor = tensor.view(3, 112, 112)
                else:
                    raise ValueError(f"Unexpected tensor shape: {tensor.shape}")
            else:
                raise ValueError(f"Unexpected tensor dimensions: {tensor.dim()}, shape: {tensor.shape}")
            
            # Ensure the tensor is the right shape [3, 112, 112]
            if tensor.shape != (3, 112, 112):
                print(f"Warning: Tensor shape {tensor.shape} from {tensor_path}, creating dummy tensor")
                tensor = torch.randn(3, 112, 112, device=self.device)
            
            return tensor
            
        except Exception as e:
            print(f"Error loading tensor {tensor_path}: {e}")
            # Return a dummy tensor if loading fails
            return torch.randn(3, 112, 112, device=self.device)
    
    def _extract_face_parts(self, image_path):
        """Extract face parts from image"""
        try:
            # Get face landmarks
            face_landmarks = get_position(image_path)
            if face_landmarks is None:
                print(f"Could not detect landmarks in {image_path}")
                return self._get_dummy_face_parts()
            
            # Extract all face parts
            face_parts_tensors = {}
            
            for part_name, part_indices in self.face_parts.items():
                try:
                    # Get face part region
                    corners = square_from_side_midpoints(
                        face_landmarks[part_indices[0]], 
                        face_landmarks[part_indices[1]]
                    )
                    
                    # Create temporary output path
                    temp_path = f"/tmp/temp_{part_name}.jpg"
                    
                    # Extract face part
                    warp_perspective_cutout(image_path, corners, temp_path, output_size=(224, 224))
                    
                    # Load and transform
                    if os.path.exists(temp_path):
                        image = Image.open(temp_path).convert('RGB')
                        face_parts_tensors[part_name] = self.face_parts_transform(image)
                        
                        # Clean up temporary file
                        os.remove(temp_path)
                    else:
                        face_parts_tensors[part_name] = torch.zeros(3, 224, 224)
                        
                except Exception as e:
                    print(f"Error extracting {part_name}: {e}")
                    face_parts_tensors[part_name] = torch.zeros(3, 224, 224)
            
            return face_parts_tensors
            
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            return self._get_dummy_face_parts()
    
    def _get_dummy_face_parts(self):
        """Create dummy face parts tensors"""
        return {
            part_name: torch.zeros(3, 224, 224) 
            for part_name in self.face_parts.keys()
        }
    
    def get_embedding(self, image_path, tensor_path):
        """
        Get embedding vector for a single image
        
        Args:
            image_path: Path to the JPG image
            tensor_path: Path to the .pt tensor file for AdaFace
            
        Returns:
            embedding: Numpy array of shape (embedding_dim,) - the final embedding
                      Default: 512D, configurable (256D, 512D, 768D, or 1152D)
        """
        with torch.no_grad():
            # Load AdaFace tensor
            adaface_tensor = self._load_tensor_file(tensor_path)
            adaface_tensor = adaface_tensor.unsqueeze(0).to(self.device)  # Add batch dimension
            
            # Extract face parts
            face_parts_tensors = self._extract_face_parts(image_path)
            
            # Move to device and add batch dimension
            face_parts_batch = {}
            for part_name, tensor in face_parts_tensors.items():
                face_parts_batch[part_name] = tensor.unsqueeze(0).to(self.device)
            
            # Get embedding from model
            final_features, combined_features, global_features, local_features = self.model(
                adaface_tensor, face_parts_batch
            )
            
            # Return final embedding as numpy array
            return final_features.cpu().numpy().flatten()
    
    def get_batch_embedding(self, image_paths, tensor_paths):
        """
        Get embeddings for a batch of images
        
        Args:
            image_paths: List of image paths
            tensor_paths: List of tensor paths (must match image_paths)
            
        Returns:
            embeddings: Numpy array of shape (batch_size, embedding_dim)
                       Default embedding_dim: 512D, configurable
        """
        if len(image_paths) != len(tensor_paths):
            raise ValueError("Number of image paths must match number of tensor paths")
        
        embeddings = []
        for img_path, tensor_path in zip(image_paths, tensor_paths):
            embedding = self.get_embedding(img_path, tensor_path)
            embeddings.append(embedding)
        
        return np.stack(embeddings)

# Convenience functions for backward compatibility
def load_twin_model(model_path, device=None):
    """Load the twin verification model"""
    return TwinInference(model_path, device)

def get_embedding(model, image_path, tensor_path):
    """Get embedding for a single image"""
    return model.get_embedding(image_path, tensor_path)

def get_batch_embedding(model, image_paths, tensor_paths):
    """Get embeddings for a batch of images"""
    return model.get_batch_embedding(image_paths, tensor_paths)

# Test function
def test_inference():
    """Test the inference function"""
    print("Testing twin inference...")
    
    # Check if model exists
    model_path = "twin_verification_training/checkpoints/best.pth"
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        print("Please train the model first using train_twin_verification.py")
        return
    
    # Load model
    inference_model = TwinInference(model_path)
    
    # Test with dummy data (you can replace with real paths)
    test_image_path = "path/to/test/image.jpg"
    test_tensor_path = "path/to/test/tensor.pt"
    
    if os.path.exists(test_image_path) and os.path.exists(test_tensor_path):
        embedding = inference_model.get_embedding(test_image_path, test_tensor_path)
        print(f"Embedding shape: {embedding.shape}")
        print(f"Embedding (first 5 values): {embedding[:5]}")
    else:
        print("Test files not found, skipping actual inference test")
    
    print("Twin inference test completed!")

if __name__ == "__main__":
    test_inference() 