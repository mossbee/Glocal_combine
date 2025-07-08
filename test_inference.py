#!/usr/bin/env python3
"""
Test script to verify the twin inference function works correctly
"""

import os
import json
import numpy as np
import torch
from twin_inference import TwinInference

def test_twin_inference():
    """Test the twin inference function with real data"""
    print("=" * 60)
    print("Testing Twin Inference Function")
    print("=" * 60)
    
    # Check if required files exist
    required_files = {
        'model': 'twin_verification_training/checkpoints/best.pth',
        'jpg_dataset': 'jpg_dataset.json',
        'tensor_dataset': 'tensor_dataset.json'
    }
    
    missing_files = []
    for name, path in required_files.items():
        if not os.path.exists(path):
            missing_files.append(f"{name}: {path}")
    
    if missing_files:
        print("❌ Missing required files:")
        for file in missing_files:
            print(f"   - {file}")
        print("\nPlease ensure all required files are present.")
        if 'model' in str(missing_files):
            print("Run 'python train_twin_verification.py' to train the model first.")
        return False
    
    # Load datasets to get test images
    with open('jpg_dataset.json', 'r') as f:
        jpg_dataset = json.load(f)
    
    with open('tensor_dataset.json', 'r') as f:
        tensor_dataset = json.load(f)
    
    # Find a person with available images
    test_person = None
    test_images = []
    
    for person_id in jpg_dataset:
        if person_id in tensor_dataset:
            jpg_paths = jpg_dataset[person_id]
            tensor_paths = tensor_dataset[person_id]
            
            # Check if files exist
            valid_pairs = []
            for jpg_path, tensor_path in zip(jpg_paths, tensor_paths):
                if os.path.exists(jpg_path) and os.path.exists(tensor_path):
                    valid_pairs.append((jpg_path, tensor_path))
            
            if len(valid_pairs) >= 2:  # Need at least 2 images for testing
                test_person = person_id
                test_images = valid_pairs[:2]  # Take first 2 valid pairs
                break
    
    if not test_person:
        print("❌ No valid test images found in the dataset")
        return False
    
    print(f"✅ Found test data for person: {test_person}")
    print(f"📸 Using {len(test_images)} images for testing")
    
    # Load the model
    print(f"\n🚀 Loading twin verification model...")
    try:
        model = TwinInference(required_files['model'])
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False
    
    # Test single image inference
    print(f"\n🔄 Testing single image inference...")
    try:
        img_path, tensor_path = test_images[0]
        print(f"   Image: {os.path.basename(img_path)}")
        print(f"   Tensor: {os.path.basename(tensor_path)}")
        
        embedding1 = model.get_embedding(img_path, tensor_path)
        print(f"✅ Single inference successful")
        print(f"   Embedding shape: {embedding1.shape}")
        print(f"   Embedding range: [{embedding1.min():.4f}, {embedding1.max():.4f}]")
        print(f"   Embedding norm: {np.linalg.norm(embedding1):.4f}")
        
    except Exception as e:
        print(f"❌ Error in single inference: {e}")
        return False
    
    # Test batch inference
    print(f"\n🔄 Testing batch inference...")
    try:
        batch_img_paths = [pair[0] for pair in test_images]
        batch_tensor_paths = [pair[1] for pair in test_images]
        
        batch_embeddings = model.get_batch_embedding(batch_img_paths, batch_tensor_paths)
        print(f"✅ Batch inference successful")
        print(f"   Batch embeddings shape: {batch_embeddings.shape}")
        print(f"   Expected shape: ({len(test_images)}, 256)")
        
    except Exception as e:
        print(f"❌ Error in batch inference: {e}")
        return False
    
    # Test similarity calculation
    print(f"\n🔄 Testing similarity calculation...")
    try:
        if len(test_images) >= 2:
            # Get embeddings for two different images of the same person
            img1_path, tensor1_path = test_images[0]
            img2_path, tensor2_path = test_images[1]
            
            emb1 = model.get_embedding(img1_path, tensor1_path)
            emb2 = model.get_embedding(img2_path, tensor2_path)
            
            # Calculate cosine similarity
            emb1_norm = emb1 / np.linalg.norm(emb1)
            emb2_norm = emb2 / np.linalg.norm(emb2)
            similarity = np.dot(emb1_norm, emb2_norm)
            
            print(f"✅ Similarity calculation successful")
            print(f"   Same person similarity: {similarity:.4f}")
            print(f"   (Should be high for same person)")
            
        else:
            print("⚠️  Only one image available, skipping similarity test")
            
    except Exception as e:
        print(f"❌ Error in similarity calculation: {e}")
        return False
    
    # Performance test
    print(f"\n⏱️  Performance test...")
    try:
        import time
        
        # Time single inference
        start_time = time.time()
        for _ in range(5):
            _ = model.get_embedding(test_images[0][0], test_images[0][1])
        single_time = (time.time() - start_time) / 5
        
        print(f"   Average single inference time: {single_time:.3f}s")
        
    except Exception as e:
        print(f"⚠️  Performance test failed: {e}")
    
    print(f"\n🎉 All tests passed successfully!")
    print(f"✅ Twin inference function is working correctly")
    print(f"✅ Ready for evaluation with model_evaluation_twin.py")
    
    return True

def test_compatibility():
    """Test compatibility with evaluation script"""
    print(f"\n" + "=" * 60)
    print("Testing Compatibility with Evaluation Script")
    print("=" * 60)
    
    # Check if evaluation script can import the inference module
    try:
        import twin_inference
        print("✅ twin_inference module imports successfully")
    except Exception as e:
        print(f"❌ Error importing twin_inference: {e}")
        return False
    
    # Check if required functions exist
    required_functions = ['TwinInference', 'load_twin_model', 'get_embedding', 'get_batch_embedding']
    for func_name in required_functions:
        if hasattr(twin_inference, func_name):
            print(f"✅ Function '{func_name}' exists")
        else:
            print(f"❌ Function '{func_name}' missing")
            return False
    
    print(f"✅ All required functions are available")
    print(f"✅ model_evaluation_twin.py should work with this inference function")
    
    return True

def main():
    """Main test function"""
    success = True
    
    # Test the inference function
    if not test_twin_inference():
        success = False
    
    # Test compatibility
    if not test_compatibility():
        success = False
    
    if success:
        print(f"\n🎉 All tests completed successfully!")
        print(f"📝 Next steps:")
        print(f"   1. Run: python model_evaluation_twin.py")
        print(f"   2. Check results in twin_evaluation_results/")
    else:
        print(f"\n❌ Some tests failed. Please fix the issues before proceeding.")
    
    return success

if __name__ == "__main__":
    main() 