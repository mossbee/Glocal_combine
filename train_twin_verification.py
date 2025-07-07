import os
import time
import json
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm

from combined_model import TwinVerificationModel, TripletLoss
from triplet_dataset import create_data_loaders
from face_parts_extractor import FacePartsExtractor

class TwinVerificationTrainer:
    """Complete training pipeline for twin verification"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Create output directories
        os.makedirs(config['output_dir'], exist_ok=True)
        os.makedirs(os.path.join(config['output_dir'], 'checkpoints'), exist_ok=True)
        os.makedirs(os.path.join(config['output_dir'], 'logs'), exist_ok=True)
        
        # Initialize tensorboard
        self.writer = SummaryWriter(os.path.join(config['output_dir'], 'logs'))
        
        # Initialize model
        self._setup_model()
        
        # Initialize data loaders
        self._setup_data_loaders()
        
        # Initialize optimizer and scheduler
        self._setup_optimizer()
        
        # Initialize loss function
        self.criterion = TripletLoss(margin=config['triplet_margin'])
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
    
    def _setup_model(self):
        """Initialize the twin verification model"""
        self.model = TwinVerificationModel(
            adaface_arch=self.config['adaface_arch'],
            face_parts_embedding_dim=self.config['face_parts_embedding_dim'],
            freeze_adaface=self.config['freeze_adaface']
        )
        self.model.to(self.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
    
    def _setup_data_loaders(self):
        """Initialize train and validation data loaders"""
        self.train_loader, self.val_loader = create_data_loaders(
            tensor_dataset_path=self.config['tensor_dataset_path'],
            face_parts_dataset_path=self.config['face_parts_dataset_path'],
            twin_pairs_path=self.config['twin_pairs_path'],
            batch_size=self.config['batch_size'],
            num_workers=self.config['num_workers'],
            train_split=self.config['train_split']
        )
        
        print(f"Train batches: {len(self.train_loader)}")
        print(f"Validation batches: {len(self.val_loader)}")
    
    def _setup_optimizer(self):
        """Initialize optimizer and learning rate scheduler"""
        # Only optimize face parts model parameters (AdaFace is frozen)
        self.optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )
        
        # Learning rate scheduler
        if self.config['scheduler'] == 'step':
            self.scheduler = StepLR(
                self.optimizer, 
                step_size=self.config['step_size'], 
                gamma=self.config['gamma']
            )
        elif self.config['scheduler'] == 'cosine':
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=self.config['num_epochs'],
                eta_min=self.config['learning_rate'] * 0.01
            )
        else:
            self.scheduler = None
    
    def _save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_loss': self.best_val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'config': self.config
        }
        
        # Save latest checkpoint
        checkpoint_path = os.path.join(self.config['output_dir'], 'checkpoints', 'latest.pth')
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = os.path.join(self.config['output_dir'], 'checkpoints', 'best.pth')
            torch.save(checkpoint, best_path)
            print(f"New best model saved with validation loss: {self.best_val_loss:.4f}")
    
    def _load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            if self.scheduler and checkpoint['scheduler_state_dict']:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            self.current_epoch = checkpoint['epoch']
            self.best_val_loss = checkpoint['best_val_loss']
            self.train_losses = checkpoint['train_losses']
            self.val_losses = checkpoint['val_losses']
            
            print(f"Loaded checkpoint from epoch {self.current_epoch}")
            return True
        return False
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        # Keep AdaFace in eval mode since it's frozen
        if self.config['freeze_adaface']:
            self.model.adaface_model.eval()
        
        total_loss = 0.0
        num_batches = len(self.train_loader)
        
        # Regenerate triplets for this epoch
        self.train_loader.dataset.regenerate_triplets()
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}/{self.config['num_epochs']}")
        
        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            anchor_tensors = batch['anchor']['tensor'].to(self.device)
            positive_tensors = batch['positive']['tensor'].to(self.device)
            negative_tensors = batch['negative']['tensor'].to(self.device)
            
            anchor_parts = {k: v.to(self.device) for k, v in batch['anchor']['face_parts'].items()}
            positive_parts = {k: v.to(self.device) for k, v in batch['positive']['face_parts'].items()}
            negative_parts = {k: v.to(self.device) for k, v in batch['negative']['face_parts'].items()}
            
            # Forward pass
            self.optimizer.zero_grad()
            
            # Get features for anchor, positive, negative
            anchor_features, _, _, _ = self.model(anchor_tensors, anchor_parts)
            positive_features, _, _, _ = self.model(positive_tensors, positive_parts)
            negative_features, _, _, _ = self.model(negative_tensors, negative_parts)
            
            # Compute triplet loss
            loss = self.criterion(anchor_features, positive_features, negative_features)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['grad_clip'])
            
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'avg_loss': f"{total_loss / (batch_idx + 1):.4f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.6f}"
            })
            
            # Log to tensorboard
            global_step = self.current_epoch * num_batches + batch_idx
            self.writer.add_scalar('Train/Loss_Batch', loss.item(), global_step)
            self.writer.add_scalar('Train/Learning_Rate', self.optimizer.param_groups[0]['lr'], global_step)
        
        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)
        
        return avg_loss
    
    def validate_epoch(self):
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0.0
        num_batches = len(self.val_loader)
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                # Move batch to device
                anchor_tensors = batch['anchor']['tensor'].to(self.device)
                positive_tensors = batch['positive']['tensor'].to(self.device)
                negative_tensors = batch['negative']['tensor'].to(self.device)
                
                anchor_parts = {k: v.to(self.device) for k, v in batch['anchor']['face_parts'].items()}
                positive_parts = {k: v.to(self.device) for k, v in batch['positive']['face_parts'].items()}
                negative_parts = {k: v.to(self.device) for k, v in batch['negative']['face_parts'].items()}
                
                # Forward pass
                anchor_features, _, _, _ = self.model(anchor_tensors, anchor_parts)
                positive_features, _, _, _ = self.model(positive_tensors, positive_parts)
                negative_features, _, _, _ = self.model(negative_tensors, negative_parts)
                
                # Compute loss
                loss = self.criterion(anchor_features, positive_features, negative_features)
                total_loss += loss.item()
        
        avg_loss = total_loss / num_batches
        self.val_losses.append(avg_loss)
        
        return avg_loss
    
    def train(self):
        """Main training loop"""
        print("Starting training...")
        
        # Load checkpoint if resuming
        if self.config['resume']:
            checkpoint_path = os.path.join(self.config['output_dir'], 'checkpoints', 'latest.pth')
            self._load_checkpoint(checkpoint_path)
        
        for epoch in range(self.current_epoch, self.config['num_epochs']):
            self.current_epoch = epoch
            start_time = time.time()
            
            # Train
            train_loss = self.train_epoch()
            
            # Validate
            val_loss = self.validate_epoch()
            
            # Update learning rate
            if self.scheduler:
                self.scheduler.step()
            
            # Log to tensorboard
            self.writer.add_scalar('Train/Loss_Epoch', train_loss, epoch)
            self.writer.add_scalar('Val/Loss_Epoch', val_loss, epoch)
            
            # Save checkpoint
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
            
            self._save_checkpoint(epoch, is_best)
            
            # Print epoch summary
            epoch_time = time.time() - start_time
            print(f"Epoch {epoch + 1}/{self.config['num_epochs']} - "
                  f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                  f"Time: {epoch_time:.2f}s")
            
            # Early stopping check
            if self.config['early_stopping_patience']:
                if len(self.val_losses) > self.config['early_stopping_patience']:
                    recent_losses = self.val_losses[-self.config['early_stopping_patience']:]
                    if all(loss >= self.best_val_loss for loss in recent_losses):
                        print(f"Early stopping triggered after {epoch + 1} epochs")
                        break
        
        print("Training completed!")
        self.writer.close()

def main():
    # Configuration
    config = {
        # Model parameters
        'adaface_arch': 'ir_50',
        'face_parts_embedding_dim': 128,
        'freeze_adaface': True,
        
        # Training parameters
        'num_epochs': 100,
        'batch_size': 8,
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'grad_clip': 1.0,
        'triplet_margin': 1.0,
        
        # Scheduler parameters
        'scheduler': 'cosine',  # 'step', 'cosine', or None
        'step_size': 30,
        'gamma': 0.1,
        
        # Data parameters
        'tensor_dataset_path': 'tensor_dataset.json',
        'face_parts_dataset_path': 'extracted_face_parts/face_parts_dataset.json',
        'twin_pairs_path': 'twin_pairs.json',
        'train_split': 0.8,
        'num_workers': 2,
        
        # Output and checkpointing
        'output_dir': 'twin_verification_training',
        'resume': False,
        'early_stopping_patience': 10,
    }
    
    # Create trainer and start training
    trainer = TwinVerificationTrainer(config)
    trainer.train()

if __name__ == "__main__":
    main() 