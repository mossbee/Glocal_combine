# Twin Face Verification System

A deep learning system for identical twin face verification that combines global facial features from AdaFace with local facial parts features using ResNet50.

## Architecture Overview

The system combines two types of features:
- **Global Features (512D)**: Extracted using frozen AdaFace IR-50 model from full face images
- **Local Features (5×128D=640D)**: Extracted using trainable ResNet50 from 5 face parts (left eye, right eye, nose, mouth, chin)
- **Final Features**: Combined 1152D features projected to 256D for similarity learning

## Project Structure

```
face_parts/
├── face_parts_extractor.py      # Pre-extract face parts from images
├── face_parts_model.py          # ResNet50 models for face parts
├── combined_model.py             # Combined AdaFace + Face parts model
├── triplet_dataset.py           # Dataset for triplet learning
├── train_twin_verification.py   # Main training script
├── cutout.py                     # Face parts extraction utilities
├── inference.py                  # AdaFace inference utilities
├── net.py                        # AdaFace network architecture
├── requirements.txt              # Python dependencies
├── tensor_dataset.json           # Paths to .pt files (AdaFace input)
├── jpg_dataset.json              # Paths to .jpg files (original images)
├── twin_pairs.json               # Twin pairs information
└── pretrained/
    └── adaface_ir50_ms1mv2.ckpt  # Pretrained AdaFace model
```

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure you have the AdaFace pretrained model in `pretrained/adaface_ir50_ms1mv2.ckpt`

## Usage

### Step 1: Extract Face Parts

First, pre-extract face parts from all images for faster training:

```bash
python face_parts_extractor.py
```

This will:
- Process all images listed in `jpg_dataset.json`
- Extract 5 face parts using MediaPipe landmarks
- Save extracted parts in `extracted_face_parts/` directory
- Create `extracted_face_parts/face_parts_dataset.json` mapping

### Step 2: Train the Model

Run the training pipeline:

```bash
python train_twin_verification.py
```

The training script will:
- Load the combined model (frozen AdaFace + trainable ResNet50)
- Create triplet datasets with random negative sampling
- Train using triplet loss with margin
- Save checkpoints and logs to `twin_verification_training/`

### Step 3: Monitor Training

Use TensorBoard to monitor training progress:

```bash
tensorboard --logdir twin_verification_training/logs
```

## Configuration

Edit the `config` dictionary in `train_twin_verification.py` to customize:

### Model Parameters
- `adaface_arch`: AdaFace architecture ('ir_50')
- `face_parts_embedding_dim`: Dimension of face parts features (128)
- `freeze_adaface`: Whether to freeze AdaFace weights (True)

### Training Parameters
- `num_epochs`: Number of training epochs (100)
- `batch_size`: Batch size (8)
- `learning_rate`: Initial learning rate (1e-4)
- `triplet_margin`: Margin for triplet loss (1.0)

### Data Parameters
- `train_split`: Train/validation split ratio (0.8)
- `tensor_dataset_path`: Path to tensor dataset JSON
- `face_parts_dataset_path`: Path to face parts dataset JSON
- `twin_pairs_path`: Path to twin pairs JSON

## Data Format

### tensor_dataset.json
```json
{
  "90003": [
    "/path/to/90003d1.pt",
    "/path/to/90003d2.pt"
  ],
  "90004": [
    "/path/to/90004d1.pt"
  ]
}
```

### jpg_dataset.json
```json
{
  "90003": [
    "/path/to/90003d1.jpg",
    "/path/to/90003d2.jpg"
  ],
  "90004": [
    "/path/to/90004d1.jpg"
  ]
}
```

### twin_pairs.json
```json
[
  ["90003", "90004"],
  ["90005", "90006"]
]
```

## Model Architecture Details

### AdaFace (Frozen)
- IR-50 architecture
- Input: 3×112×112 tensors
- Output: 512D global features
- Weights frozen during training

### Face Parts ResNet50 (Trainable)
- 5 separate ResNet50 models for each face part
- Input: 3×224×224 images per part
- Output: 128D features per part
- Total: 640D local features

### Combined Model
- Concatenates global (512D) + local (640D) = 1152D
- Final projection: 1152D → 512D → 256D
- Uses triplet loss for similarity learning

## Training Process

1. **Face Parts Extraction**: Pre-extract and save all face parts
2. **Triplet Generation**: Random negative sampling for each epoch
3. **Forward Pass**: Extract features using combined model
4. **Loss Computation**: Triplet loss with margin
5. **Backpropagation**: Only train ResNet50 parts (AdaFace frozen)
6. **Validation**: Monitor validation loss for early stopping

## Output

Training produces:
- `twin_verification_training/checkpoints/best.pth`: Best model checkpoint
- `twin_verification_training/checkpoints/latest.pth`: Latest checkpoint
- `twin_verification_training/logs/`: TensorBoard logs
- Model can be loaded for inference or further fine-tuning

## Inference

To use the trained model for twin verification:

```python
from combined_model import TwinVerificationModel
import torch

# Load trained model
model = TwinVerificationModel()
checkpoint = torch.load('twin_verification_training/checkpoints/best.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Extract features for two face images
features1, _, _, _ = model(face_tensor1, face_parts_dict1)
features2, _, _, _ = model(face_tensor2, face_parts_dict2)

# Compute similarity
similarity = torch.nn.functional.cosine_similarity(features1, features2)
```

## Hardware Requirements

- GPU with at least 8GB VRAM recommended
- 16GB+ RAM for data loading
- SSD storage for faster data access

## Notes

- The system is specifically designed for twin verification (same/different person)
- Random negative sampling provides diverse training data
- Face parts extraction requires MediaPipe landmarks detection
- AdaFace weights remain frozen to preserve pretrained knowledge
- Only ResNet50 face parts models are trained for twin-specific features 