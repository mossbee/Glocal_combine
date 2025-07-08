# Twin Face Verification System

This project implements a twin face verification system that combines global features from AdaFace with local face parts features for improved identical twin identification.

## Two-Stage Training Architecture

The system uses a **curriculum learning approach** with two distinct training stages:

### Stage 1: General Face Discrimination (Random Negatives)
- **Purpose**: Learn general face discrimination between different people
- **Negative Sampling**: Random selection from all available people (easier negatives)
- **Focus**: Building robust global and local feature representations
- **Typical Duration**: 60% of total training epochs

### Stage 2: Twin-Specific Learning (Hard Negatives)
- **Purpose**: Learn fine-grained features to distinguish identical twins
- **Negative Sampling**: Specifically use twin siblings as negatives (hard negatives)
- **Focus**: Refining features for twin discrimination
- **Typical Duration**: 40% of total training epochs

This approach follows the principle that **easier examples should be learned first**, allowing the model to establish general face understanding before tackling the challenging twin discrimination task.

## Architecture Details

### Model Components
- **Global Features**: 512D from frozen AdaFace IR-50 model
- **Local Features**: 5×128D (640D total) from trainable ResNet50 face parts models
  - Left eye, right eye, nose, mouth, chin regions
- **Combined Features**: 1152D → configurable final embedding
- **Final Embedding**: Configurable dimension (512D recommended for twin verification)
  - **1152D**: No compression, maximum feature retention (best for challenging cases)
  - **768D**: Minimal compression, high quality
  - **512D**: Standard dimension, good balance (default)
  - **256D**: Heavy compression, not recommended for twins
- **Loss Function**: Triplet loss with margin-based learning

### Face Parts Extraction
Uses MediaPipe face landmarks to extract and align face parts:
- **Left Eye**: Landmarks 35-168
- **Right Eye**: Landmarks 168-265
- **Nose**: Landmarks 36-266
- **Mouth**: Landmarks 61-291
- **Chin**: Landmarks 32-262

Each part is extracted as a 224×224 image and processed by individual ResNet50 models.

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Extract Face Parts (Required for Training)
```bash
python run_extraction.py
```
This processes all images in your dataset and creates face parts in `extracted_face_parts/`.

### 3. Two-Stage Training
```bash
python train_twin_verification.py
```

The training automatically handles both stages:
- **Stage 1 (0-60 epochs)**: Random negative sampling for general discrimination
- **Stage 2 (61-100 epochs)**: Twin negative sampling for hard negative mining

### 4. Monitor Training Progress
```bash
tensorboard --logdir twin_verification_training/logs
```

Key metrics to monitor:
- **Train/Twin_Negative_Ratio**: Shows the proportion of twin negatives (0.0 in Stage 1, high in Stage 2)
- **Train/Current_Stage**: Current training stage (1 or 2)
- **Stage-specific losses**: Separate tracking for each stage

### 5. Test Inference
```bash
python test_inference.py
```

### 6. Evaluate Model
```bash
python model_evaluation_twin.py
```

## Training Configuration

### Two-Stage Parameters
```python
config = {
    # Model architecture
    'final_embedding_dim': 512,     # Embedding dimension (512D recommended)
                                   # Options: 1152, 768, 512, 256
    
    # Two-Stage Training
    'stage_1_epochs': 60,           # General discrimination phase
    'stage_2_epochs': 40,           # Twin-specific phase
    
    # Stage 2 fine-tuning
    'stage_2_lr_factor': 0.1,       # Reduce LR for stage 2 (optional)
    'reset_scheduler_stage_2': False, # Reset scheduler for stage 2
    
    # Training parameters
    'batch_size': 8,
    'learning_rate': 1e-4,
    'triplet_margin': 1.0,
}
```

### Negative Sampling Strategies
- **'random'**: Stage 1 - Random negatives from all people
- **'twin'**: Stage 2 - Twin siblings as negatives
- **'mixed'**: 50/50 mix (for experimentation)

### Resume Training
```python
config['resume'] = True  # Resumes from latest checkpoint with correct stage
```

## Advanced Usage

### Custom Stage Configuration
```python
# Quick stage 1 → long stage 2
config.update({
    'stage_1_epochs': 30,
    'stage_2_epochs': 70,
    'stage_2_lr_factor': 0.05,  # More aggressive LR reduction
})

# Equal stages
config.update({
    'stage_1_epochs': 50,
    'stage_2_epochs': 50,
})
```

### Manual Stage Control
```python
# Start directly from stage 2 (if you have a pre-trained stage 1 model)
trainer = TwinVerificationTrainer(config)
trainer.current_stage = 2
trainer._setup_data_loaders(negative_strategy='twin')
trainer.train()
```

## File Structure

### Training Files
- `train_twin_verification.py` - Two-stage training script
- `combined_model.py` - Complete model architecture
- `triplet_dataset.py` - Dataset with stage-aware negative sampling
- `face_parts_model.py` - ResNet50 face parts models

### Inference Files
- `twin_inference.py` - Inference with both image and tensor inputs
- `model_evaluation_twin.py` - Comprehensive evaluation for twin verification
- `test_inference.py` - Test inference functionality

### Preprocessing
- `face_parts_extractor.py` - MediaPipe-based face parts extraction
- `run_extraction.py` - User-friendly extraction script

### Legacy Files
- `inference.py` - Original AdaFace-only inference
- `model_evaluation.py` - Original evaluation (incompatible with two-stage model)

## Checkpoints and Resume

The system saves multiple checkpoints:
- `latest.pth` - Most recent checkpoint with stage information
- `best.pth` - Best overall validation loss
- `stage_1_complete.pth` - Stage 1 completion checkpoint
- `stage_2_complete.pth` - Final model checkpoint

Resume automatically detects the current stage and continues appropriately.

## Evaluation Metrics

### Performance Indicators
- **EER (Equal Error Rate)**: Primary metric for twin verification
- **Stage-specific best losses**: Track improvement within each stage
- **Twin negative ratio**: Monitor hard negative mining effectiveness
- **AUC**: Area under ROC curve

### Expected Performance Patterns
- **Stage 1**: Rapid initial improvement, plateaus as general features are learned
- **Stage 2**: Slower but steady improvement as twin-specific features develop
- **Overall**: Stage 2 should achieve lower final EER than stage 1 plateau

## Tips for Best Results

### Stage Duration
- **More twin data**: Increase stage 2 duration
- **Limited twin data**: Focus on longer stage 1 for general features
- **Large dataset**: Equal stages often work well

### Learning Rate Strategy
- **Conservative**: `stage_2_lr_factor = 0.1` for fine-tuning
- **Aggressive**: `stage_2_lr_factor = 0.01` for minimal changes
- **Reset scheduler**: Use for completely independent stage training

### Data Requirements
- **Minimum**: 2+ images per person, 1+ twin pair
- **Recommended**: 5+ images per person, 3+ twin pairs
- **Optimal**: 10+ images per person, 5+ twin pairs

### Embedding Dimension Selection
Choose the right embedding dimension for your use case:

- **1152D (No Compression)**
  - Best: When you have sufficient computational resources
  - Best: For the most challenging twin pairs
  - Best: When you need maximum discriminative power
  - Cons: Higher memory usage, slower inference

- **768D (Minimal Compression)**
  - Good balance between quality and efficiency
  - Suitable for most twin verification tasks
  - Moderate computational requirements

- **512D (Standard, Recommended)**
  - **Default choice** for twin verification
  - Good balance of performance and efficiency
  - Compatible with most face recognition systems
  - Standard dimension in face recognition literature

- **256D (Heavy Compression)**
  - Only for severely resource-constrained environments
  - **Not recommended** for twin verification
  - May lose important discriminative features

```python
# Configure embedding dimension in training
config['final_embedding_dim'] = 512   # Recommended
config['final_embedding_dim'] = 1152  # Best quality
config['final_embedding_dim'] = 768   # High quality
config['final_embedding_dim'] = 256   # Not recommended
```

## Troubleshooting

### Common Issues
1. **Low twin negative ratio in stage 2**: Check twin_pairs.json format
2. **No improvement in stage 2**: Try reducing learning rate further
3. **Memory issues**: Reduce batch size or face parts resolution
4. **Stage switching errors**: Ensure proper checkpoint format

### Debugging
```bash
# Check dataset statistics
python -c "
from triplet_dataset import TwinTripletDataset
dataset = TwinTripletDataset('tensor_dataset.json', 'extracted_face_parts/face_parts_dataset.json', 'twin_pairs.json', negative_strategy='twin')
print(dataset.get_strategy_statistics())
"
```

This two-stage approach significantly improves twin verification performance by first establishing general face discrimination capabilities, then refining the model with challenging twin-specific examples.

## System Requirements
- CUDA-capable GPU (recommended)
- 8GB+ GPU memory for batch_size=8
- Python 3.8+
- See `requirements.txt` for complete dependencies 