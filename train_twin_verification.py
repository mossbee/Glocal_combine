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
    """Complete training pipeline for twin verification with two-stage training"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Training stages
        self.current_stage = 1
        self.stage_1_epochs = config.get('stage_1_epochs', config['num_epochs'] // 2)
        self.stage_2_epochs = config.get('stage_2_epochs', config['num_epochs'] - self.stage_1_epochs)
        self.total_epochs = self.stage_1_epochs + self.stage_2_epochs
        
        print(f"Two-Stage Training Configuration:")
        print(f"  Stage 1 (Random negatives): {self.stage_1_epochs} epochs")
        print(f"  Stage 2 (Twin negatives): {self.stage_2_epochs} epochs")
        print(f"  Total epochs: {self.total_epochs}")
        
        # Create output directories
        os.makedirs(config['output_dir'], exist_ok=True)
        os.makedirs(os.path.join(config['output_dir'], 'checkpoints'), exist_ok=True)
        os.makedirs(os.path.join(config['output_dir'], 'logs'), exist_ok=True)
        
        # Initialize tensorboard
        self.writer = SummaryWriter(os.path.join(config['output_dir'], 'logs'))
        
        # Initialize model
        self._setup_model()
        
        # Initialize data loaders (will be updated when switching stages)
        self._setup_data_loaders()
        
        # Initialize optimizer and scheduler
        self._setup_optimizer()
        
        # Initialize loss function
        self.criterion = TripletLoss(margin=config['triplet_margin'])
        
        # Training state
        self.current_epoch = 0
        self.stage_start_epoch = 0
        self.best_val_loss = float('inf')
        self.best_stage_1_loss = float('inf')
        self.best_stage_2_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        
        # Stage-specific tracking
        self.stage_1_losses = []
        self.stage_2_losses = []
    
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
    
    def _setup_data_loaders(self, negative_strategy='random'):
        """Initialize train and validation data loaders with specified strategy"""
        self.train_loader, self.val_loader = create_data_loaders(
            tensor_dataset_path=self.config['tensor_dataset_path'],
            face_parts_dataset_path=self.config['face_parts_dataset_path'],
            twin_pairs_path=self.config['twin_pairs_path'],
            batch_size=self.config['batch_size'],
            num_workers=self.config['num_workers'],
            train_split=self.config['train_split'],
            negative_strategy=negative_strategy
        )
        
        print(f"Data loaders setup with negative_strategy='{negative_strategy}'")
        print(f"Train batches: {len(self.train_loader)}")
        print(f"Validation batches: {len(self.val_loader)}")
        
        # Print dataset statistics
        if hasattr(self.train_loader.dataset, 'get_strategy_statistics'):
            stats = self.train_loader.dataset.get_strategy_statistics()
            print(f"Training dataset statistics: {stats}")
    
    def _switch_to_stage_2(self):
        """Switch from Stage 1 (random negatives) to Stage 2 (twin negatives)"""
        print("\n" + "="*60)
        print("SWITCHING TO STAGE 2: TWIN HARD NEGATIVES")
        print("="*60)
        
        self.current_stage = 2
        self.stage_start_epoch = self.current_epoch
        
        # Update data loaders with twin negative strategy
        self._setup_data_loaders(negative_strategy='twin')
        
        # Optionally adjust learning rate for stage 2
        if self.config.get('stage_2_lr_factor', None):
            new_lr = self.config['learning_rate'] * self.config['stage_2_lr_factor']
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = new_lr
            print(f"Adjusted learning rate for Stage 2: {new_lr}")
        
        # Reset scheduler for stage 2 if configured
        if self.config.get('reset_scheduler_stage_2', False):
            self._setup_optimizer(stage_2=True)
            print("Reset optimizer and scheduler for Stage 2")
        
        # Save stage 1 completion checkpoint
        self._save_checkpoint(self.current_epoch - 1, is_stage_completion=True, stage=1)
        
        print(f"Stage 2 will run for {self.stage_2_epochs} epochs")
        print("Focus: Learning to distinguish identical twins")
    
    def _setup_optimizer(self, stage_2=False):
        """Initialize optimizer and learning rate scheduler"""
        # Only optimize face parts model parameters (AdaFace is frozen)
        lr = self.config['learning_rate']
        if stage_2 and self.config.get('stage_2_lr_factor', None):
            lr *= self.config['stage_2_lr_factor']
            
        self.optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=lr,
            weight_decay=self.config['weight_decay']
        )
        
        # Learning rate scheduler
        epochs_for_scheduler = self.total_epochs
        if stage_2 and self.config.get('reset_scheduler_stage_2', False):
            epochs_for_scheduler = self.stage_2_epochs
            
        if self.config['scheduler'] == 'step':
            self.scheduler = StepLR(
                self.optimizer, 
                step_size=self.config['step_size'], 
                gamma=self.config['gamma']
            )
        elif self.config['scheduler'] == 'cosine':
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=epochs_for_scheduler,
                eta_min=lr * 0.01
            )
        else:
            self.scheduler = None
    
    def _save_checkpoint(self, epoch, is_best=False, is_stage_completion=False, stage=None):
        """Save model checkpoint with stage information"""
        checkpoint = {
            'epoch': epoch,
            'current_stage': self.current_stage,
            'stage_start_epoch': self.stage_start_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_loss': self.best_val_loss,
            'best_stage_1_loss': self.best_stage_1_loss,
            'best_stage_2_loss': self.best_stage_2_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'stage_1_losses': self.stage_1_losses,
            'stage_2_losses': self.stage_2_losses,
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
        
        # Save stage completion checkpoint
        if is_stage_completion and stage:
            stage_path = os.path.join(self.config['output_dir'], 'checkpoints', f'stage_{stage}_complete.pth')
            torch.save(checkpoint, stage_path)
            print(f"Stage {stage} completion checkpoint saved")
    
    def _load_checkpoint(self, checkpoint_path):
        """Load model checkpoint with stage information"""
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            if self.scheduler and checkpoint['scheduler_state_dict']:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            self.current_epoch = checkpoint['epoch']
            self.current_stage = checkpoint.get('current_stage', 1)
            self.stage_start_epoch = checkpoint.get('stage_start_epoch', 0)
            self.best_val_loss = checkpoint['best_val_loss']
            self.best_stage_1_loss = checkpoint.get('best_stage_1_loss', float('inf'))
            self.best_stage_2_loss = checkpoint.get('best_stage_2_loss', float('inf'))
            self.train_losses = checkpoint['train_losses']
            self.val_losses = checkpoint['val_losses']
            self.stage_1_losses = checkpoint.get('stage_1_losses', [])
            self.stage_2_losses = checkpoint.get('stage_2_losses', [])
            
            print(f"Loaded checkpoint from epoch {self.current_epoch}, stage {self.current_stage}")
            
            # Setup data loaders based on current stage
            if self.current_stage == 1:
                self._setup_data_loaders(negative_strategy='random')
            else:
                self._setup_data_loaders(negative_strategy='twin')
                
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
        
        # Get current strategy statistics
        stats = self.train_loader.dataset.get_strategy_statistics()
        
        stage_name = "Stage 1 (Random)" if self.current_stage == 1 else "Stage 2 (Twin)"
        epoch_in_stage = self.current_epoch - self.stage_start_epoch + 1
        max_epochs_in_stage = self.stage_1_epochs if self.current_stage == 1 else self.stage_2_epochs
        
        pbar = tqdm(self.train_loader, 
                   desc=f"{stage_name} - Epoch {epoch_in_stage}/{max_epochs_in_stage} "
                        f"(Twin ratio: {stats.get('twin_negative_ratio', 0):.2f})")
        
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
                'lr': f"{self.optimizer.param_groups[0]['lr']:.6f}",
                'stage': self.current_stage
            })
            
            # Log to tensorboard
            global_step = self.current_epoch * num_batches + batch_idx
            self.writer.add_scalar('Train/Loss_Batch', loss.item(), global_step)
            self.writer.add_scalar('Train/Learning_Rate', self.optimizer.param_groups[0]['lr'], global_step)
            self.writer.add_scalar('Train/Stage', self.current_stage, global_step)
            self.writer.add_scalar('Train/Twin_Negative_Ratio', stats.get('twin_negative_ratio', 0), global_step)
        
        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)
        
        # Track stage-specific losses
        if self.current_stage == 1:
            self.stage_1_losses.append(avg_loss)
        else:
            self.stage_2_losses.append(avg_loss)
        
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
        """Main training loop with two-stage training"""
        print("Starting two-stage training...")
        print(f"Stage 1: Learning general face discrimination ({self.stage_1_epochs} epochs)")
        print(f"Stage 2: Learning twin-specific features ({self.stage_2_epochs} epochs)")
        
        # Load checkpoint if resuming
        if self.config['resume']:
            checkpoint_path = os.path.join(self.config['output_dir'], 'checkpoints', 'latest.pth')
            self._load_checkpoint(checkpoint_path)
        
        for epoch in range(self.current_epoch, self.total_epochs):
            self.current_epoch = epoch
            start_time = time.time()
            
            # Check if we need to switch to stage 2
            if self.current_stage == 1 and epoch >= self.stage_1_epochs:
                self._switch_to_stage_2()
            
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
            self.writer.add_scalar('Train/Current_Stage', self.current_stage, epoch)
            
            # Track stage-specific best losses
            if self.current_stage == 1:
                is_stage_best = val_loss < self.best_stage_1_loss
                if is_stage_best:
                    self.best_stage_1_loss = val_loss
            else:
                is_stage_best = val_loss < self.best_stage_2_loss
                if is_stage_best:
                    self.best_stage_2_loss = val_loss
            
            # Save checkpoint
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
            
            self._save_checkpoint(epoch, is_best)
            
            # Print epoch summary
            epoch_time = time.time() - start_time
            stage_info = f"Stage {self.current_stage}"
            epoch_in_stage = epoch - self.stage_start_epoch + 1
            max_epochs_in_stage = self.stage_1_epochs if self.current_stage == 1 else self.stage_2_epochs
            
            print(f"Epoch {epoch + 1}/{self.total_epochs} ({stage_info}: {epoch_in_stage}/{max_epochs_in_stage}) - "
                  f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                  f"Time: {epoch_time:.2f}s")
            
            # Early stopping check (only within current stage)
            if self.config['early_stopping_patience']:
                stage_losses = self.stage_1_losses if self.current_stage == 1 else self.stage_2_losses
                stage_best = self.best_stage_1_loss if self.current_stage == 1 else self.best_stage_2_loss
                
                if len(stage_losses) > self.config['early_stopping_patience']:
                    recent_losses = stage_losses[-self.config['early_stopping_patience']:]
                    if all(loss >= stage_best for loss in recent_losses):
                        print(f"Early stopping triggered in Stage {self.current_stage} after {epoch + 1} epochs")
                        if self.current_stage == 1:
                            print("Proceeding to Stage 2...")
                            # Force switch to stage 2
                            self.stage_1_epochs = epoch + 1
                            continue
                        else:
                            break
        
        # Save final stage completion
        if self.current_stage == 2:
            self._save_checkpoint(self.current_epoch, is_stage_completion=True, stage=2)
        
        print("\nTwo-stage training completed!")
        print(f"Best Stage 1 loss: {self.best_stage_1_loss:.4f}")
        print(f"Best Stage 2 loss: {self.best_stage_2_loss:.4f}")
        print(f"Overall best loss: {self.best_val_loss:.4f}")
        self.writer.close()

def main():
    # Configuration for Two-Stage Training
    config = {
        # Model parameters
        'adaface_arch': 'ir_50',
        'face_parts_embedding_dim': 128,
        'freeze_adaface': True,
        
        # Two-Stage Training parameters
        'num_epochs': 100,  # Total epochs (will be split between stages)
        'stage_1_epochs': 60,  # Stage 1: Random negatives (general discrimination)
        'stage_2_epochs': 40,  # Stage 2: Twin negatives (hard negatives)
        
        # Stage 2 specific parameters
        'stage_2_lr_factor': 0.1,  # Reduce learning rate for stage 2 (optional)
        'reset_scheduler_stage_2': False,  # Whether to reset scheduler for stage 2
        
        # Training parameters
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
    
    print("=" * 80)
    print("TWO-STAGE TWIN VERIFICATION TRAINING")
    print("=" * 80)
    print(f"Stage 1: Random negatives (general face discrimination) - {config['stage_1_epochs']} epochs")
    print(f"Stage 2: Twin negatives (hard negative mining) - {config['stage_2_epochs']} epochs")
    print(f"Total epochs: {config['stage_1_epochs'] + config['stage_2_epochs']}")
    print("=" * 80)
    
    # Validate configuration
    if config['stage_1_epochs'] + config['stage_2_epochs'] != config['num_epochs']:
        print("Warning: stage_1_epochs + stage_2_epochs != num_epochs")
        print("Adjusting num_epochs to match stage epochs...")
        config['num_epochs'] = config['stage_1_epochs'] + config['stage_2_epochs']
    
    # Create trainer and start training
    trainer = TwinVerificationTrainer(config)
    trainer.train()

if __name__ == "__main__":
    main() 