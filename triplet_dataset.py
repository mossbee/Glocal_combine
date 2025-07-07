import os
import json
import random
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import numpy as np

class TwinTripletDataset(Dataset):
    """Dataset for triplet learning with twin verification"""
    
    def __init__(self, tensor_dataset_path, face_parts_dataset_path, twin_pairs_path, 
                 mode='train', train_split=0.8, face_parts_transform=None):
        """
        Initialize the triplet dataset
        
        Args:
            tensor_dataset_path: Path to tensor_dataset.json (.pt files for AdaFace)
            face_parts_dataset_path: Path to face_parts_dataset.json (extracted face parts)
            twin_pairs_path: Path to twin_pairs.json
            mode: 'train' or 'val' 
            train_split: Ratio for train/val split
            face_parts_transform: Transform for face parts images
        """
        self.mode = mode
        
        # Load datasets
        with open(tensor_dataset_path, 'r') as f:
            self.tensor_dataset = json.load(f)
            
        with open(face_parts_dataset_path, 'r') as f:
            self.face_parts_dataset = json.load(f)
            
        with open(twin_pairs_path, 'r') as f:
            self.twin_pairs = json.load(f)
        
        # Face parts names
        self.part_names = ['left_eye', 'right_eye', 'nose', 'mouth', 'chin']
        
        # Set up transforms
        if face_parts_transform is None:
            self.face_parts_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])
        else:
            self.face_parts_transform = face_parts_transform
        
        # Create train/val split
        self._create_train_val_split(train_split)
        
        # Create person ID lists for negative sampling
        self._setup_person_ids()
        
        # Generate triplets for the current epoch
        self._generate_triplets()
    
    def _create_train_val_split(self, train_split):
        """Split twin pairs into train and validation sets"""
        random.shuffle(self.twin_pairs)
        n_train = int(len(self.twin_pairs) * train_split)
        
        if self.mode == 'train':
            self.current_pairs = self.twin_pairs[:n_train]
        else:
            self.current_pairs = self.twin_pairs[n_train:]
    
    def _setup_person_ids(self):
        """Set up person IDs available in current split"""
        self.person_ids = set()
        for pair in self.current_pairs:
            self.person_ids.update(pair)
        self.person_ids = list(self.person_ids)
    
    def _generate_triplets(self, num_triplets_per_person=3):
        """Generate triplets for the current epoch"""
        self.triplets = []
        
        for person_id in self.person_ids:
            # Get available images for this person
            if person_id not in self.tensor_dataset:
                continue
                
            person_images = self.tensor_dataset[person_id]
            if len(person_images) < 2:
                continue  # Need at least 2 images for anchor/positive
            
            # Generate multiple triplets for this person
            for _ in range(num_triplets_per_person):
                # Choose anchor and positive from same person
                anchor_img, positive_img = random.sample(person_images, 2)
                
                # Choose negative from different person (random sampling)
                negative_person = random.choice([pid for pid in self.person_ids if pid != person_id])
                if negative_person in self.tensor_dataset and len(self.tensor_dataset[negative_person]) > 0:
                    negative_img = random.choice(self.tensor_dataset[negative_person])
                    
                    self.triplets.append({
                        'anchor': (person_id, anchor_img),
                        'positive': (person_id, positive_img),
                        'negative': (negative_person, negative_img)
                    })
    
    def _load_tensor_file(self, tensor_path):
        """Load .pt tensor file for AdaFace"""
        try:
            tensor = torch.load(tensor_path)
            
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
                tensor = torch.randn(3, 112, 112)
            
            return tensor
            
        except Exception as e:
            print(f"Error loading tensor {tensor_path}: {e}")
            # Return a dummy tensor if loading fails
            return torch.randn(3, 112, 112)
    
    def _load_face_parts(self, person_id, image_name):
        """Load face parts for a given person and image"""
        face_parts = {}
        
        # Get image name without extension for lookup
        if image_name.endswith('.pt'):
            lookup_name = image_name.replace('.pt', '')
        else:
            lookup_name = os.path.splitext(os.path.basename(image_name))[0]
        
        # Load each face part
        for part_name in self.part_names:
            try:
                if (person_id in self.face_parts_dataset and 
                    lookup_name in self.face_parts_dataset[person_id] and
                    part_name in self.face_parts_dataset[person_id][lookup_name]):
                    
                    part_path = self.face_parts_dataset[person_id][lookup_name][part_name]
                    
                    if os.path.exists(part_path):
                        # Load and transform face part image
                        image = Image.open(part_path).convert('RGB')
                        face_parts[part_name] = self.face_parts_transform(image)
                    else:
                        # Create dummy tensor if image doesn't exist
                        face_parts[part_name] = torch.zeros(3, 224, 224)
                else:
                    # Create dummy tensor if part not found
                    face_parts[part_name] = torch.zeros(3, 224, 224)
                    
            except Exception as e:
                print(f"Error loading face part {part_name} for {person_id}/{lookup_name}: {e}")
                face_parts[part_name] = torch.zeros(3, 224, 224)
        
        return face_parts
    
    def __len__(self):
        return len(self.triplets)
    
    def __getitem__(self, idx):
        """Get a triplet sample"""
        triplet = self.triplets[idx]
        
        # Load anchor
        anchor_person, anchor_path = triplet['anchor']
        anchor_tensor = self._load_tensor_file(anchor_path)
        anchor_parts = self._load_face_parts(anchor_person, anchor_path)
        
        # Load positive
        positive_person, positive_path = triplet['positive']
        positive_tensor = self._load_tensor_file(positive_path)
        positive_parts = self._load_face_parts(positive_person, positive_path)
        
        # Load negative  
        negative_person, negative_path = triplet['negative']
        negative_tensor = self._load_tensor_file(negative_path)
        negative_parts = self._load_face_parts(negative_person, negative_path)
        
        return {
            'anchor': {
                'tensor': anchor_tensor,
                'face_parts': anchor_parts,
                'person_id': anchor_person
            },
            'positive': {
                'tensor': positive_tensor,
                'face_parts': positive_parts,
                'person_id': positive_person
            },
            'negative': {
                'tensor': negative_tensor,
                'face_parts': negative_parts,
                'person_id': negative_person
            }
        }
    
    def regenerate_triplets(self, num_triplets_per_person=3):
        """Regenerate triplets for a new epoch"""
        self._generate_triplets(num_triplets_per_person)

def collate_triplets(batch):
    """Custom collate function for triplet batch"""
    batch_size = len(batch)
    
    # Initialize tensors
    anchor_tensors = torch.stack([item['anchor']['tensor'] for item in batch])
    positive_tensors = torch.stack([item['positive']['tensor'] for item in batch])
    negative_tensors = torch.stack([item['negative']['tensor'] for item in batch])
    
    # Initialize face parts dictionaries
    part_names = ['left_eye', 'right_eye', 'nose', 'mouth', 'chin']
    
    anchor_parts = {}
    positive_parts = {}
    negative_parts = {}
    
    for part_name in part_names:
        anchor_parts[part_name] = torch.stack([item['anchor']['face_parts'][part_name] for item in batch])
        positive_parts[part_name] = torch.stack([item['positive']['face_parts'][part_name] for item in batch])
        negative_parts[part_name] = torch.stack([item['negative']['face_parts'][part_name] for item in batch])
    
    # Person IDs
    anchor_ids = [item['anchor']['person_id'] for item in batch]
    positive_ids = [item['positive']['person_id'] for item in batch]
    negative_ids = [item['negative']['person_id'] for item in batch]
    
    return {
        'anchor': {
            'tensor': anchor_tensors,
            'face_parts': anchor_parts,
            'person_ids': anchor_ids
        },
        'positive': {
            'tensor': positive_tensors,
            'face_parts': positive_parts,
            'person_ids': positive_ids
        },
        'negative': {
            'tensor': negative_tensors,
            'face_parts': negative_parts,
            'person_ids': negative_ids
        }
    }

def create_data_loaders(tensor_dataset_path, face_parts_dataset_path, twin_pairs_path,
                       batch_size=8, num_workers=2, train_split=0.8):
    """Create train and validation data loaders"""
    
    # Create datasets
    train_dataset = TwinTripletDataset(
        tensor_dataset_path=tensor_dataset_path,
        face_parts_dataset_path=face_parts_dataset_path,
        twin_pairs_path=twin_pairs_path,
        mode='train',
        train_split=train_split
    )
    
    val_dataset = TwinTripletDataset(
        tensor_dataset_path=tensor_dataset_path,
        face_parts_dataset_path=face_parts_dataset_path,
        twin_pairs_path=twin_pairs_path,
        mode='val',
        train_split=train_split
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_triplets,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_triplets,
        pin_memory=True
    )
    
    return train_loader, val_loader

def test_dataset():
    """Test the triplet dataset"""
    dataset = TwinTripletDataset(
        tensor_dataset_path="tensor_dataset.json",
        face_parts_dataset_path="extracted_face_parts/face_parts_dataset.json",
        twin_pairs_path="twin_pairs.json",
        mode='train'
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Test a sample
    sample = dataset[0]
    print(f"Sample keys: {sample.keys()}")
    print(f"Anchor tensor shape: {sample['anchor']['tensor'].shape}")
    print(f"Anchor face parts keys: {sample['anchor']['face_parts'].keys()}")
    print(f"Face part shape: {sample['anchor']['face_parts']['left_eye'].shape}")

if __name__ == "__main__":
    test_dataset() 