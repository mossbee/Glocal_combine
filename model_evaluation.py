import os
import csv
import json
import torch
import numpy as np
from tqdm import tqdm
import inference
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from itertools import combinations, product

# data folder 
ROOT_IMAGE_DIR = '/media/sdc/nhat.nh/Dataset/ND_TWIN_AdaFace'
# json file with info about id and images
INFOR_JSON = '/media/sdc/nhat.nh/Dataset/AdaFace_Info/AdaFace_yaw_0_pairs_test.json'
# json file with info on twin pair 
TWIN_JSON = '/media/sdc/nhat.nh/Dataset/pairs_test.json'
# output folder for all metrics and csv files
OUTPUT_DIR = '/media/sdc/nhat.nh/Projects/Identical_Twin_Verification/AdaFace/evaluation_results/AdaFace_yaw_0_pairs_test'

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

BATCH_SIZE = 64
os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(INFOR_JSON, 'r') as f:
    person2images = json.load(f)

with open(TWIN_JSON, 'r') as f:
    twin_pairs = json.load(f)

all_img_paths = []
img2person = dict()
for person_id, img_list in person2images.items():
    for img_name in img_list:
        img_path = os.path.join(ROOT_IMAGE_DIR, person_id, img_name)
        all_img_paths.append(img_path)
        img2person[img_path] = person_id

model = inference.load_pretrained_model(architecture='ir_101', device=DEVICE)
# model = image_model_handler.load_tuned_model(architecture='ir_101', model_path='/media/sdc/nhat.nh/Projects/Identical_Twin_Verification/AdaFace/pretrained/best_model.pth', device=DEVICE)

embeddings = []
for i in tqdm(range(0, len(all_img_paths), BATCH_SIZE), desc="Extracting embeddings"):
    batch_paths = all_img_paths[i:i+BATCH_SIZE]
    batch_embeds = inference.get_batch_embedding(model, batch_paths, device=DEVICE)  # (batch, 512) on CUDA
    embeddings.append(batch_embeds.detach().cpu().numpy())  # ensure on CPU for later use

embeddings = np.vstack(embeddings)  # shape (N_images, 512)
img2embedding = dict(zip(all_img_paths, embeddings))

genuine_pairs_info = []
print("Calculating genuine (same person) pairs...")
for person_id, img_list in tqdm(person2images.items(), desc="Genuine pairs"):
    img_paths = [os.path.join(ROOT_IMAGE_DIR, person_id, img) for img in img_list]
    embs = np.stack([img2embedding[img] for img in img_paths], axis=0)
    embs_norm = embs / np.linalg.norm(embs, axis=1, keepdims=True)
    sim_matrix = embs_norm @ embs_norm.T
    upper_indices = np.triu_indices(len(embs), k=1)
    scores = sim_matrix[upper_indices]
    pairs = [(img_paths[i], img_paths[j]) for i, j in zip(*upper_indices)]
    for (img1, img2), score in zip(pairs, scores.tolist()):
        genuine_pairs_info.append((img1, img2, score, 1))

twin_impostor_pairs_info = []
print("Calculating twin impostor pairs...")
for twin_id1, twin_id2 in tqdm(twin_pairs, desc="Twin impostor pairs"):
    if twin_id1 not in person2images or twin_id2 not in person2images:
        print(f"Warning: {twin_id1} or {twin_id2} not in dataset, skipping twin pair.")
        continue
    imgs1 = [os.path.join(ROOT_IMAGE_DIR, twin_id1, img) for img in person2images[twin_id1]]
    imgs2 = [os.path.join(ROOT_IMAGE_DIR, twin_id2, img) for img in person2images[twin_id2]]
    embs1 = np.stack([img2embedding[img] for img in imgs1], axis=0)
    embs2 = np.stack([img2embedding[img] for img in imgs2], axis=0)
    embs1_norm = embs1 / np.linalg.norm(embs1, axis=1, keepdims=True)
    embs2_norm = embs2 / np.linalg.norm(embs2, axis=1, keepdims=True)
    sim_matrix = embs1_norm @ embs2_norm.T  # Shape (N1,N2)
    for i in range(sim_matrix.shape[0]):
        for j in range(sim_matrix.shape[1]):
            twin_impostor_pairs_info.append(
                (imgs1[i], imgs2[j], sim_matrix[i, j], 0)
            )

all_pairs_info = genuine_pairs_info + twin_impostor_pairs_info

# ------------ EVALUATION (as before, but now threshold is used for ACC) ------------

all_scores = np.array([x[2] for x in all_pairs_info], dtype=np.float32)
all_labels = np.array([x[3] for x in all_pairs_info], dtype=np.int32)

fpr, tpr, thresholds = roc_curve(all_labels, all_scores, pos_label=1)
fnr = 1 - tpr
eer_idx = np.nanargmin(np.abs(fpr - fnr))
eer_threshold = thresholds[eer_idx]
eer = fpr[eer_idx]
FAR = fpr[eer_idx]
FRR = fnr[eer_idx]

print(f"Equal Error Rate (EER): {eer*100:.2f}%")
print(f"EER Threshold (cosine): {eer_threshold:.4f}")
print(f"FAR at EER: {FAR*100:.2f}%")
print(f"FRR at EER: {FRR*100:.2f}%")

# ------------- COMPUTE CLASSIFICATION ACCURACY AT BEST THRESHOLD -------------
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

# ----- Save metrics -----
metrics = {
    "EER": float(eer),
    "EER_Threshold": float(eer_threshold),
    "FAR_at_EER": float(FAR),
    "FRR_at_EER": float(FRR),
    "Accuracy_At_EER_Threshold": float(accuracy),
}

with open(os.path.join(OUTPUT_DIR, "metrics.json"), "w") as f:
    json.dump(metrics, f, indent=4)

# -------- Save ROC curve as before ---------
plt.figure()
plt.plot(fpr, tpr, label="ROC curve")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(OUTPUT_DIR, 'roc_curve.png'))
plt.close()

# ---------- Save detailed CSV: [img1, img2, score, label, pred] -----------
csv_path = os.path.join(OUTPUT_DIR, "scores_labels_pairs.csv")
with open(csv_path, "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['img1', 'img2', 'score', 'label', 'prediction'])
    for row in pairs_for_csv:
        writer.writerow(row)