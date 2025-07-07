import os
import json
import cv2
import numpy as np
from tqdm import tqdm
from cutout import get_position, square_from_side_midpoints, warp_perspective_cutout

class FacePartsExtractor:
    def __init__(self, jpg_dataset_path, output_dir):
        """
        Extract and save face parts from JPG images
        
        Args:
            jpg_dataset_path: Path to jpg_dataset.json
            output_dir: Directory to save extracted face parts
        """
        self.jpg_dataset_path = jpg_dataset_path
        self.output_dir = output_dir
        
        # Face parts configuration from cutout.py
        self.face_parts = {
            "left_eye": [35, 168],
            "right_eye": [168, 265], 
            "mouth": [61, 291],
            "nose": [36, 266],
            "chin": [32, 262]
        }
        
        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        for part in self.face_parts.keys():
            os.makedirs(os.path.join(output_dir, part), exist_ok=True)
    
    def extract_face_part(self, image_path, face_landmarks, part_name, part_indices):
        """Extract a single face part using landmarks"""
        try:
            corners = square_from_side_midpoints(
                face_landmarks[part_indices[0]], 
                face_landmarks[part_indices[1]]
            )
            
            # Create output path
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            output_path = os.path.join(self.output_dir, part_name, f"{base_name}_{part_name}.jpg")
            
            # Extract and save face part
            warp_perspective_cutout(image_path, corners, output_path, output_size=(224, 224))
            return output_path
            
        except Exception as e:
            print(f"Error extracting {part_name} from {image_path}: {e}")
            return None
    
    def process_image(self, image_path):
        """Process a single image and extract all face parts"""
        # Get face landmarks
        face_landmarks = get_position(image_path)
        if face_landmarks is None:
            print(f"Could not detect landmarks in {image_path}")
            return {}
        
        # Extract all face parts
        extracted_parts = {}
        for part_name, part_indices in self.face_parts.items():
            part_path = self.extract_face_part(image_path, face_landmarks, part_name, part_indices)
            if part_path:
                extracted_parts[part_name] = part_path
        
        return extracted_parts
    
    def extract_all_face_parts(self):
        """Extract face parts from all images in the dataset"""
        # Load dataset
        with open(self.jpg_dataset_path, 'r') as f:
            jpg_dataset = json.load(f)
        
        # Create mapping of extracted face parts
        face_parts_dataset = {}
        
        print("Extracting face parts from all images...")
        for person_id, image_paths in tqdm(jpg_dataset.items()):
            face_parts_dataset[person_id] = {}
            
            for image_path in image_paths:
                if not os.path.exists(image_path):
                    print(f"Image not found: {image_path}")
                    continue
                
                # Extract face parts for this image
                extracted_parts = self.process_image(image_path)
                
                # Store paths using image filename as key
                image_name = os.path.splitext(os.path.basename(image_path))[0]
                face_parts_dataset[person_id][image_name] = extracted_parts
        
        # Save face parts dataset mapping
        output_json = os.path.join(self.output_dir, "face_parts_dataset.json")
        with open(output_json, 'w') as f:
            json.dump(face_parts_dataset, f, indent=2)
        
        print(f"Face parts extraction completed! Dataset saved to {output_json}")
        return face_parts_dataset

def main():
    extractor = FacePartsExtractor(
        jpg_dataset_path="jpg_dataset.json",
        output_dir="extracted_face_parts"
    )
    
    face_parts_dataset = extractor.extract_all_face_parts()
    print("Face parts extraction pipeline completed!")

if __name__ == "__main__":
    main() 