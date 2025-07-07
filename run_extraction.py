#!/usr/bin/env python3
"""
Simple script to run face parts extraction pipeline
This should be run before training to pre-extract all face parts
"""

import os
import sys
from face_parts_extractor import FacePartsExtractor

def main():
    print("=" * 60)
    print("Twin Face Verification - Face Parts Extraction")
    print("=" * 60)
    
    # Check if required files exist
    required_files = [
        'jpg_dataset.json',
        'face_landmarker_v2_with_blendshapes.task'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print("❌ Missing required files:")
        for file in missing_files:
            print(f"   - {file}")
        print("\nPlease ensure all required files are present before running extraction.")
        return 1
    
    print("✅ All required files found")
    print(f"📁 JPG Dataset: jpg_dataset.json")
    print(f"🎯 MediaPipe Model: face_landmarker_v2_with_blendshapes.task")
    
    # Create extractor
    print("\n🚀 Initializing Face Parts Extractor...")
    try:
        extractor = FacePartsExtractor(
            jpg_dataset_path="jpg_dataset.json",
            output_dir="extracted_face_parts"
        )
        
        print("✅ Extractor initialized successfully")
        print(f"📂 Output directory: extracted_face_parts/")
        
        # Run extraction
        print("\n🔄 Starting face parts extraction...")
        print("This may take a while depending on the number of images...")
        
        face_parts_dataset = extractor.extract_all_face_parts()
        
        print("\n🎉 Face parts extraction completed successfully!")
        print(f"📋 Dataset mapping saved to: extracted_face_parts/face_parts_dataset.json")
        print(f"👀 Extracted parts: left_eye, right_eye, nose, mouth, chin")
        
        # Print summary
        total_persons = len(face_parts_dataset)
        total_images = sum(len(images) for images in face_parts_dataset.values())
        
        print(f"\n📊 Extraction Summary:")
        print(f"   - Total persons: {total_persons}")
        print(f"   - Total images processed: {total_images}")
        print(f"   - Face parts per image: 5")
        print(f"   - Total face parts extracted: {total_images * 5}")
        
        print("\n✅ Ready for training! Run 'python train_twin_verification.py' next.")
        
    except Exception as e:
        print(f"\n❌ Error during extraction: {e}")
        print("Please check the error message and try again.")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 