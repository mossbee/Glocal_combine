import os
import csv
import json
import torch
import numpy as np
from tqdm import tqdm
import twin_inference
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from itertools import combinations, product

# Configuration - Update these paths for your setup
CONFIG = {
    # Model path
    'model_path': 'twin_verification_training/checkpoints/best.pth',
    
    # Data paths - Update these to match your dataset structure
    'jpg_dataset_path': 'jpg_dataset.json',
    'tensor_dataset_path': 'tensor_dataset.json', 
    'twin_pairs_path': 'twin_pairs.json',
    
    # Output directory
    'output_dir': 'twin_evaluation_results',
    
    # Processing parameters
    'batch_size': 8,  # Smaller batch size for the combined model
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

print(f"Using device: {CONFIG['device']}")

# Create output directory
os.makedirs(CONFIG['output_dir'], exist_ok=True)

# Load datasets
print("Loading datasets...")
with open(CONFIG['jpg_dataset_path'], 'r') as f:
    jpg_dataset = json.load(f)

with open(CONFIG['tensor_dataset_path'], 'r') as f:
    tensor_dataset = json.load(f)

with open(CONFIG['twin_pairs_path'], 'r') as f:
    twin_pairs = json.load(f)

# Create mappings
print("Creating image-tensor mappings...")
image_tensor_pairs = []
person_to_pairs = {}

for person_id in jpg_dataset:
    if person_id not in tensor_dataset:
        print(f"Warning: Person {person_id} not found in tensor dataset")
        continue
    
    jpg_paths = jpg_dataset[person_id]
    tensor_paths = tensor_dataset[person_id]
    
    if len(jpg_paths) != len(tensor_paths):
        print(f"Warning: Mismatched number of images for person {person_id}")
        continue
    
    person_pairs = []
    for jpg_path, tensor_path in zip(jpg_paths, tensor_paths):
        if os.path.exists(jpg_path) and os.path.exists(tensor_path):
            pair = (jpg_path, tensor_path)
            image_tensor_pairs.append(pair)
            person_pairs.append(pair)
        else:
            print(f"Warning: Missing files for {person_id}: {jpg_path} or {tensor_path}")
    
    if person_pairs:
        person_to_pairs[person_id] = person_pairs

print(f"Total valid image-tensor pairs: {len(image_tensor_pairs)}")
print(f"Total persons with valid pairs: {len(person_to_pairs)}")

# Load the twin verification model
print(f"Loading twin verification model from {CONFIG['model_path']}")
if not os.path.exists(CONFIG['model_path']):
    print(f"Error: Model file not found at {CONFIG['model_path']}")
    print("Please train the model first using train_twin_verification.py")
    exit(1)

model = twin_inference.TwinInference(CONFIG['model_path'], device=CONFIG['device'])

# Extract embeddings
print("Extracting embeddings...")
embeddings = []
image_tensor_mapping = {}

for i in tqdm(range(0, len(image_tensor_pairs), CONFIG['batch_size']), desc="Extracting embeddings"):
    batch_pairs = image_tensor_pairs[i:i+CONFIG['batch_size']]
    
    # Separate image and tensor paths
    batch_image_paths = [pair[0] for pair in batch_pairs]
    batch_tensor_paths = [pair[1] for pair in batch_pairs]
    
    # Get batch embeddings
    try:
        batch_embeddings = model.get_batch_embedding(batch_image_paths, batch_tensor_paths)
        embeddings.append(batch_embeddings)
        
        # Create mapping for later use
        for j, (img_path, tensor_path) in enumerate(batch_pairs):
            image_tensor_mapping[img_path] = batch_embeddings[j]
    except Exception as e:
        print(f"Error processing batch {i}: {e}")
        # Create dummy embeddings for failed batch
        dummy_embeddings = np.zeros((len(batch_pairs), 256))
        embeddings.append(dummy_embeddings)
        for j, (img_path, tensor_path) in enumerate(batch_pairs):
            image_tensor_mapping[img_path] = dummy_embeddings[j]

embeddings = np.vstack(embeddings)  # shape (N_images, 256)
print(f"Embeddings shape: {embeddings.shape}")

# Generate genuine pairs (same person)
print("Calculating genuine (same person) pairs...")
genuine_pairs_info = []

for person_id, pairs in tqdm(person_to_pairs.items(), desc="Genuine pairs"):
    if len(pairs) < 2:
        continue  # Need at least 2 images for pairs
    
    # Get embeddings for this person
    person_embeddings = []
    person_image_paths = []
    
    for img_path, tensor_path in pairs:
        if img_path in image_tensor_mapping:
            person_embeddings.append(image_tensor_mapping[img_path])
            person_image_paths.append(img_path)
    
    if len(person_embeddings) < 2:
        continue
    
    # Calculate pairwise similarities
    embs = np.stack(person_embeddings, axis=0)
    embs_norm = embs / np.linalg.norm(embs, axis=1, keepdims=True)
    sim_matrix = embs_norm @ embs_norm.T
    
    # Get upper triangular indices (avoid duplicates)
    upper_indices = np.triu_indices(len(embs), k=1)
    scores = sim_matrix[upper_indices]
    
    # Create pairs
    for idx_i, idx_j in zip(*upper_indices):
        img1 = person_image_paths[idx_i]
        img2 = person_image_paths[idx_j]
        score = sim_matrix[idx_i, idx_j]
        genuine_pairs_info.append((img1, img2, score, 1))

print(f"Generated {len(genuine_pairs_info)} genuine pairs")

# Generate twin impostor pairs (different twin siblings)
print("Calculating twin impostor pairs...")
twin_impostor_pairs_info = []

for twin_id1, twin_id2 in tqdm(twin_pairs, desc="Twin impostor pairs"):
    if twin_id1 not in person_to_pairs or twin_id2 not in person_to_pairs:
        print(f"Warning: {twin_id1} or {twin_id2} not in dataset, skipping twin pair.")
        continue
    
    pairs1 = person_to_pairs[twin_id1]
    pairs2 = person_to_pairs[twin_id2]
    
    # Get embeddings for both twins
    embs1 = []
    imgs1 = []
    for img_path, tensor_path in pairs1:
        if img_path in image_tensor_mapping:
            embs1.append(image_tensor_mapping[img_path])
            imgs1.append(img_path)
    
    embs2 = []
    imgs2 = []
    for img_path, tensor_path in pairs2:
        if img_path in image_tensor_mapping:
            embs2.append(image_tensor_mapping[img_path])
            imgs2.append(img_path)
    
    if len(embs1) == 0 or len(embs2) == 0:
        continue
    
    # Calculate cross-similarities
    embs1 = np.stack(embs1, axis=0)
    embs2 = np.stack(embs2, axis=0)
    embs1_norm = embs1 / np.linalg.norm(embs1, axis=1, keepdims=True)
    embs2_norm = embs2 / np.linalg.norm(embs2, axis=1, keepdims=True)
    
    sim_matrix = embs1_norm @ embs2_norm.T  # Shape (N1, N2)
    
    # Create all pairs between the two twins
    for i in range(sim_matrix.shape[0]):
        for j in range(sim_matrix.shape[1]):
            img1 = imgs1[i]
            img2 = imgs2[j]
            score = sim_matrix[i, j]
            twin_impostor_pairs_info.append((img1, img2, score, 0))

print(f"Generated {len(twin_impostor_pairs_info)} twin impostor pairs")

# Combine all pairs
all_pairs_info = genuine_pairs_info + twin_impostor_pairs_info
print(f"Total pairs for evaluation: {len(all_pairs_info)}")

if len(all_pairs_info) == 0:
    print("Error: No pairs generated for evaluation!")
    exit(1)

# Extract scores and labels
all_scores = np.array([x[2] for x in all_pairs_info], dtype=np.float32)
all_labels = np.array([x[3] for x in all_pairs_info], dtype=np.int32)

print(f"Score statistics:")
print(f"  Min: {all_scores.min():.4f}")
print(f"  Max: {all_scores.max():.4f}")
print(f"  Mean: {all_scores.mean():.4f}")
print(f"  Std: {all_scores.std():.4f}")
print(f"Genuine pairs: {np.sum(all_labels == 1)}")
print(f"Impostor pairs: {np.sum(all_labels == 0)}")

# Calculate ROC curve and EER
fpr, tpr, thresholds = roc_curve(all_labels, all_scores, pos_label=1)
fnr = 1 - tpr
eer_idx = np.nanargmin(np.abs(fpr - fnr))
eer_threshold = thresholds[eer_idx]
eer = fpr[eer_idx]
FAR = fpr[eer_idx]
FRR = fnr[eer_idx]

print(f"\nEvaluation Results:")
print(f"Equal Error Rate (EER): {eer*100:.2f}%")
print(f"EER Threshold (cosine): {eer_threshold:.4f}")
print(f"FAR at EER: {FAR*100:.2f}%")
print(f"FRR at EER: {FRR*100:.2f}%")

# Calculate classification accuracy at EER threshold
correct_pred = 0
all_pred = 0
pairs_for_csv = []

for img1, img2, score, label in all_pairs_info:
    # Predict: 1 if score >= threshold else 0
    pred = 1 if score >= eer_threshold else 0
    if pred == label:
        correct_pred += 1
    all_pred += 1
    pairs_for_csv.append([img1, img2, score, label, pred])

accuracy = correct_pred / all_pred
print(f"Classification Accuracy at EER Threshold: {accuracy*100:.2f}%")

# Calculate AUC
from sklearn.metrics import auc
auc_score = auc(fpr, tpr)
print(f"AUC Score: {auc_score:.4f}")

# Save metrics
metrics = {
    "EER": float(eer),
    "EER_Threshold": float(eer_threshold),
    "FAR_at_EER": float(FAR),
    "FRR_at_EER": float(FRR),
    "Accuracy_At_EER_Threshold": float(accuracy),
    "AUC": float(auc_score),
    "Total_Pairs": len(all_pairs_info),
    "Genuine_Pairs": int(np.sum(all_labels == 1)),
    "Impostor_Pairs": int(np.sum(all_labels == 0))
}

with open(os.path.join(CONFIG['output_dir'], "metrics.json"), "w") as f:
    json.dump(metrics, f, indent=4)

print(f"\nMetrics saved to {os.path.join(CONFIG['output_dir'], 'metrics.json')}")

# Save ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc_score:.3f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Twin Verification')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(CONFIG['output_dir'], 'roc_curve.png'), dpi=300, bbox_inches='tight')
plt.close()

print(f"ROC curve saved to {os.path.join(CONFIG['output_dir'], 'roc_curve.png')}")

# Save detailed results
csv_path = os.path.join(CONFIG['output_dir'], "detailed_results.csv")
with open(csv_path, "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['img1', 'img2', 'score', 'label', 'prediction'])
    for row in pairs_for_csv:
        writer.writerow(row)

print(f"Detailed results saved to {csv_path}")
print("\nEvaluation completed successfully!") 