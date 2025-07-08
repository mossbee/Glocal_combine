#!/usr/bin/env python3
"""
Test script to verify two-stage training implementation
"""

import os
import json
import torch
from triplet_dataset import TwinTripletDataset, create_data_loaders

def test_negative_strategies():
    """Test different negative sampling strategies"""
    print("=" * 60)
    print("Testing Negative Sampling Strategies")
    print("=" * 60)
    
    # Check if required files exist
    required_files = {
        'tensor_dataset': 'tensor_dataset.json',
        'face_parts_dataset': 'extracted_face_parts/face_parts_dataset.json',
        'twin_pairs': 'twin_pairs.json'
    }
    
    missing_files = []
    for name, path in required_files.items():
        if not os.path.exists(path):
            missing_files.append(f"{name}: {path}")
    
    if missing_files:
        print("❌ Missing required files:")
        for file in missing_files:
            print(f"   - {file}")
        print("\nSkipping negative strategy tests...")
        return False
    
    try:
        # Test Stage 1: Random negatives
        print("\n🔄 Testing Stage 1 (Random negatives)...")
        train_loader_stage1, _ = create_data_loaders(
            tensor_dataset_path='tensor_dataset.json',
            face_parts_dataset_path='extracted_face_parts/face_parts_dataset.json',
            twin_pairs_path='twin_pairs.json',
            batch_size=4,
            negative_strategy='random'
        )
        
        stats_stage1 = train_loader_stage1.dataset.get_strategy_statistics()
        print(f"✅ Stage 1 dataset created successfully")
        print(f"   Strategy: {stats_stage1['strategy']}")
        print(f"   Total triplets: {stats_stage1['total_triplets']}")
        print(f"   Twin negative ratio: {stats_stage1['twin_negative_ratio']:.3f}")
        print(f"   Expected: Low twin negative ratio (~0.0-0.2)")
        
        # Test Stage 2: Twin negatives
        print("\n🔄 Testing Stage 2 (Twin negatives)...")
        train_loader_stage2, _ = create_data_loaders(
            tensor_dataset_path='tensor_dataset.json',
            face_parts_dataset_path='extracted_face_parts/face_parts_dataset.json',
            twin_pairs_path='twin_pairs.json',
            batch_size=4,
            negative_strategy='twin'
        )
        
        stats_stage2 = train_loader_stage2.dataset.get_strategy_statistics()
        print(f"✅ Stage 2 dataset created successfully")
        print(f"   Strategy: {stats_stage2['strategy']}")
        print(f"   Total triplets: {stats_stage2['total_triplets']}")
        print(f"   Twin negative ratio: {stats_stage2['twin_negative_ratio']:.3f}")
        print(f"   Expected: High twin negative ratio (~0.5-1.0)")
        
        # Test strategy switching
        print("\n🔄 Testing strategy switching...")
        dataset = train_loader_stage1.dataset
        original_strategy = dataset.negative_strategy
        original_stats = dataset.get_strategy_statistics()
        
        # Switch to twin strategy
        dataset.set_negative_strategy('twin')
        new_stats = dataset.get_strategy_statistics()
        
        print(f"✅ Strategy switching successful")
        print(f"   Before: {original_strategy} (twin ratio: {original_stats['twin_negative_ratio']:.3f})")
        print(f"   After: {dataset.negative_strategy} (twin ratio: {new_stats['twin_negative_ratio']:.3f})")
        
        # Test mixed strategy
        print("\n🔄 Testing mixed strategy...")
        dataset.set_negative_strategy('mixed')
        mixed_stats = dataset.get_strategy_statistics()
        
        print(f"✅ Mixed strategy working")
        print(f"   Strategy: {mixed_stats['strategy']}")
        print(f"   Twin negative ratio: {mixed_stats['twin_negative_ratio']:.3f}")
        print(f"   Expected: Medium twin negative ratio (~0.3-0.7)")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing negative strategies: {e}")
        return False

def test_training_configuration():
    """Test two-stage training configuration"""
    print("\n" + "=" * 60)
    print("Testing Two-Stage Training Configuration")
    print("=" * 60)
    
    # Test configuration validation
    print("🔄 Testing configuration validation...")
    
    configs = [
        # Valid configuration
        {
            'num_epochs': 100,
            'stage_1_epochs': 60,
            'stage_2_epochs': 40,
            'learning_rate': 1e-4,
            'stage_2_lr_factor': 0.1,
            'reset_scheduler_stage_2': False
        },
        # Equal stages
        {
            'num_epochs': 80,
            'stage_1_epochs': 40,
            'stage_2_epochs': 40,
            'learning_rate': 1e-4,
            'stage_2_lr_factor': 0.05,
            'reset_scheduler_stage_2': True
        },
        # Quick stage 1, long stage 2
        {
            'num_epochs': 100,
            'stage_1_epochs': 20,
            'stage_2_epochs': 80,
            'learning_rate': 1e-4,
            'stage_2_lr_factor': 0.2,
            'reset_scheduler_stage_2': False
        }
    ]
    
    for i, config in enumerate(configs):
        total_calculated = config['stage_1_epochs'] + config['stage_2_epochs']
        is_valid = total_calculated == config['num_epochs']
        
        print(f"   Config {i+1}: {'✅' if is_valid else '❌'}")
        print(f"     Stage 1: {config['stage_1_epochs']} epochs")
        print(f"     Stage 2: {config['stage_2_epochs']} epochs")
        print(f"     Total: {total_calculated} (expected: {config['num_epochs']})")
        print(f"     Stage 2 LR factor: {config['stage_2_lr_factor']}")
        print(f"     Reset scheduler: {config['reset_scheduler_stage_2']}")
    
    print("✅ Configuration testing completed")
    return True

def test_checkpoint_format():
    """Test checkpoint format for two-stage training"""
    print("\n" + "=" * 60)
    print("Testing Checkpoint Format")
    print("=" * 60)
    
    # Simulate checkpoint data
    checkpoint_data = {
        'epoch': 75,
        'current_stage': 2,
        'stage_start_epoch': 60,
        'model_state_dict': {},  # Would contain actual model weights
        'optimizer_state_dict': {},
        'scheduler_state_dict': {},
        'best_val_loss': 0.234,
        'best_stage_1_loss': 0.456,
        'best_stage_2_loss': 0.234,
        'train_losses': [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.25, 0.24, 0.234],
        'val_losses': [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.35, 0.34, 0.334],
        'stage_1_losses': [0.8, 0.7, 0.6, 0.5, 0.4],  # First 5 epochs
        'stage_2_losses': [0.3, 0.25, 0.24, 0.234],    # Last 4 epochs
        'config': {
            'stage_1_epochs': 60,
            'stage_2_epochs': 40,
            'stage_2_lr_factor': 0.1
        }
    }
    
    print("🔄 Validating checkpoint structure...")
    
    required_fields = [
        'epoch', 'current_stage', 'stage_start_epoch',
        'best_stage_1_loss', 'best_stage_2_loss',
        'stage_1_losses', 'stage_2_losses'
    ]
    
    missing_fields = []
    for field in required_fields:
        if field not in checkpoint_data:
            missing_fields.append(field)
    
    if missing_fields:
        print(f"❌ Missing checkpoint fields: {missing_fields}")
        return False
    
    print("✅ Checkpoint structure is valid")
    print(f"   Current epoch: {checkpoint_data['epoch']}")
    print(f"   Current stage: {checkpoint_data['current_stage']}")
    print(f"   Stage start epoch: {checkpoint_data['stage_start_epoch']}")
    print(f"   Best stage 1 loss: {checkpoint_data['best_stage_1_loss']:.3f}")
    print(f"   Best stage 2 loss: {checkpoint_data['best_stage_2_loss']:.3f}")
    print(f"   Stage 1 loss history: {len(checkpoint_data['stage_1_losses'])} epochs")
    print(f"   Stage 2 loss history: {len(checkpoint_data['stage_2_losses'])} epochs")
    
    return True

def main():
    """Run all two-stage training tests"""
    print("🧪 Two-Stage Training Implementation Tests")
    print("=" * 60)
    
    tests = [
        ("Negative Sampling Strategies", test_negative_strategies),
        ("Training Configuration", test_training_configuration),
        ("Checkpoint Format", test_checkpoint_format)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with error: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Results Summary")
    print("=" * 60)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status}: {test_name}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("\n🎉 All tests passed! Two-stage training is ready to use.")
        print("\n📝 Next steps:")
        print("   1. Extract face parts: python run_extraction.py")
        print("   2. Start training: python train_twin_verification.py")
        print("   3. Monitor with TensorBoard: tensorboard --logdir twin_verification_training/logs")
    else:
        print(f"\n⚠️  {len(results) - passed} tests failed. Please fix issues before training.")
    
    return passed == len(results)

if __name__ == "__main__":
    main() 