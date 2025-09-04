import os
import sys
import time
import math
import gc
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from tqdm import tqdm
import wandb
import numpy as np
import yaml

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from datasets.parallel_unified_dataset import ParallelUnifiedDataset
from datasets.balanced_mixup import BalancedMixupDataset
from models.multitask_model import MultiTaskHead


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compute_lesion_class_weights(dataset: ParallelUnifiedDataset, device: torch.device) -> torch.Tensor:
    """Compute class weights for the lesion task to handle class imbalance."""
    # Count lesion labels from training dataset
    lesion_labels = []
    for idx in range(len(dataset)):
        _, labels, masks, _ = dataset[idx]
        if masks['lesion'].item() == 1:  # Only count samples with valid lesion labels
            # Harden soft labels if present (mixup)
            if isinstance(labels['lesion'], torch.Tensor) and labels['lesion'].dtype.is_floating_point and labels['lesion'].numel() > 1:
                lesion_labels.append(int(labels['lesion'].argmax().item()))
            else:
                lesion_labels.append(int(labels['lesion'].item()))
    
    # Compute class counts for 8 lesion classes
    class_counts = np.bincount(lesion_labels, minlength=8)
    
    # Compute inverse frequency weights
    weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
    weights = weights / weights.sum()  # Normalize to sum to 1
    
    # Move to device
    lesion_weights = weights.to(device)
    
    # Print class counts and weights for debugging
    print("Lesion class counts:", class_counts.tolist())
    print("Lesion class weights:", weights.tolist())
    
    return lesion_weights


class FocalLoss(nn.Module):
    def __init__(self, alpha: Optional[torch.Tensor] = None, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        import torch.nn.functional as F
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)
        loss = ((1 - pt) ** self.gamma) * ce_loss
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

class CBFocalLoss(nn.Module):
    def __init__(self, alpha: torch.Tensor, gamma: float = 2.0):
        super().__init__()
        self.register_buffer('alpha', alpha)
        self.gamma = gamma

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        import torch.nn.functional as F
        log_probs = F.log_softmax(inputs, dim=1)
        probs = torch.exp(log_probs)
        pt = probs.gather(1, targets.view(-1, 1)).squeeze(1)
        alpha_t = self.alpha.gather(0, targets)
        loss = -alpha_t * ((1 - pt) ** self.gamma) * log_probs.gather(1, targets.view(-1, 1)).squeeze(1)
        return loss.mean()

def compute_loss(outputs: Dict[str, torch.Tensor], labels: Dict[str, torch.Tensor], masks: Dict[str, torch.Tensor],
				criteria: Dict[str, nn.Module], task_weights: Dict[str, float]) -> torch.Tensor:
	"""Compute weighted sum of per-task losses with masks."""
	loss_total = 0.0
	for task in ('skin', 'lesion', 'bm'):
		mask = masks[task]  # [B]
		if mask.sum() == 0:
			continue
		# Select only valid indices
		valid_idx = mask.nonzero(as_tuple=True)[0]
		if valid_idx.numel() == 0:
			continue
		logits = outputs[task][valid_idx]
		targets = labels[task][valid_idx]
		loss = criteria[task](logits, targets)
		loss_total = loss_total + task_weights[task] * loss
	return loss_total


def train_parallel(
		experiment: str = 'exp4',
		skin_weight: float = 1.0,
		lesion_weight: float = 1.0,
		bm_weight: float = 1.0,
		hidden_dims = (512, 256),
		dropout: float = 0.3,
		batch_size: int = 32,
		epochs: int = 30,
		learning_rate: float = 1e-3,
		weight_decay: float = 1e-2,
		wandb_project: str = 'skin-lesion-classification',
		wandb_name: str = 'parallel_exp4',
		log_file: Optional[str] = None
):
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	
	# Set up file logging if log_file is provided
	if log_file:
		file_handler = logging.FileHandler(log_file)
		file_handler.setLevel(logging.INFO)
		formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
		file_handler.setFormatter(formatter)
		logger.addHandler(file_handler)
	
	logger.info(f"Using device: {device}")

	# Datasets and loaders
	train_dataset = ParallelUnifiedDataset(experiment=experiment, split='train')
	val_dataset = ParallelUnifiedDataset(experiment=experiment, split='val')

	# Optional Balanced-MixUp
	balanced_mixup_cfg = {'enabled': False, 'alpha': 0.2}
	config_path = project_root / 'config.yaml'
	if config_path.exists():
		try:
			with open(config_path, 'r') as f:
				cfg = yaml.safe_load(f) or {}
			bm = cfg.get('balanced_mixup', {}) or {}
			balanced_mixup_cfg['enabled'] = bool(bm.get('enabled', False))
			balanced_mixup_cfg['alpha'] = float(bm.get('alpha', 0.2))
		except Exception as e:
			print(f"[WARN] Failed to read balanced_mixup from config.yaml: {e}")

	if balanced_mixup_cfg['enabled']:
		print(f"Using Balanced MixUp augmentation (alpha={balanced_mixup_cfg['alpha']})")
		logger.info(f"Using Balanced MixUp augmentation (alpha={balanced_mixup_cfg['alpha']})")
		train_dataset = BalancedMixupDataset(train_dataset, num_lesion_classes=8, alpha=balanced_mixup_cfg['alpha'])

	train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
	val_loader = DataLoader(val_dataset, batch_size=batch_size*2, shuffle=False, num_workers=0, pin_memory=True)

	# Model - ensure lesion head has 8 classes for fine-grained classification (no NOT_SKIN)
	model = MultiTaskHead(
		input_dim=256, 
		hidden_dims=tuple(hidden_dims), 
		dropout=dropout,
		num_classes_skin=2,
		num_classes_lesion=8,  # 8 fine-grained lesion classes (no NOT_SKIN)
		num_classes_bm=2
	)
	model = model.to(device)

	# Compute class weights for lesion task to handle class imbalance
	base_for_counts = train_dataset.base if hasattr(train_dataset, 'base') else train_dataset
	lesion_weights = compute_lesion_class_weights(base_for_counts, device)
	
	# Read global config to select lesion loss function
	config_path = project_root / 'config.yaml'
	lesion_loss_fn = 'weighted_ce'
	focal_gamma = 2.0
	cb_beta = 0.9999
	curriculum_cfg = {'enabled': False, 'lesion_start': 1.0, 'lesion_end': 1.0, 'bm_start': 1.0, 'bm_end': 1.0, 'epochs': 20}
	if config_path.exists():
		try:
			with open(config_path, 'r') as f:
				cfg = yaml.safe_load(f) or {}
			lesion_loss_fn = str(cfg.get('lesion_loss_fn', lesion_loss_fn)).strip().lower()
			focal_gamma = float(cfg.get('focal_gamma', focal_gamma))
			cb_beta = float(cfg.get('cb_beta', cb_beta))
			cur_in = cfg.get('curriculum', {}) or {}
			if bool(cur_in.get('enabled', False)):
				curriculum_cfg.update({
					'enabled': True,
					'lesion_start': float(cur_in.get('lesion_start', 0.5)),
					'lesion_end': float(cur_in.get('lesion_end', 1.0)),
					'bm_start': float(cur_in.get('bm_start', 0.5)),
					'bm_end': float(cur_in.get('bm_end', 1.0)),
					'epochs': int(cur_in.get('epochs', 20)),
				})
		except Exception as e:
			print(f"[WARN] Failed to read config.yaml: {e}")

	# Select lesion criterion
	if lesion_loss_fn == 'focal':
		lesion_criterion = FocalLoss(alpha=lesion_weights, gamma=focal_gamma, reduction='mean')
	elif lesion_loss_fn == 'lmf':
		from losses.lmf_loss import LMFLoss
		# Compute lesion class counts from dataset to avoid hardcoding
		lesion_count_list = [0] * 8
		for i in range(len(train_dataset)):
			_, lbls, msk, _ = train_dataset[i]
			if msk['lesion'].item() == 1:
				lesion_count_list[int(lbls['lesion'].item())] += 1
		lesion_criterion = LMFLoss(cls_num_list=lesion_count_list, alpha=0.5, beta=0.5, gamma=2.0)
	elif lesion_loss_fn == 'cb_focal':
		# Compute CB alpha from counts
		lesion_labels = []
		for i in range(len(base_for_counts)):
			_, lbls, msk, _ = base_for_counts[i]
			if msk['lesion'].item() == 1:
				lesion_labels.append(int(lbls['lesion'].item()))
		counts = np.bincount(lesion_labels, minlength=8).astype(np.float64)
		effective_num = 1.0 - np.power(cb_beta, counts)
		weights_cb = (1.0 - cb_beta) / np.maximum(effective_num, 1e-12)
		weights_cb = weights_cb / weights_cb.sum()
		alpha_cb = torch.tensor(weights_cb, dtype=torch.float32, device=device)
		lesion_criterion = CBFocalLoss(alpha=alpha_cb, gamma=focal_gamma)
	else:
		lesion_criterion = nn.CrossEntropyLoss(weight=lesion_weights)

	# Log chosen lesion loss
	print(f"Using lesion_loss_fn: {lesion_loss_fn}")
	logger.info(f"Using lesion_loss_fn: {lesion_loss_fn}")
	if lesion_loss_fn == 'focal':
		print(f"Focal alpha: {lesion_weights.detach().cpu().tolist()}, gamma: 2.0")
		logger.info(f"Focal alpha: {lesion_weights.detach().cpu().tolist()}, gamma: 2.0")
	elif lesion_loss_fn == 'lmf':
		print(f"LMF loss with alpha=0.5, beta=0.5, gamma=2.0")
		logger.info(f"LMF loss with alpha=0.5, beta=0.5, gamma=2.0")

	# Losses
	criteria = {
		'skin': nn.CrossEntropyLoss(),
		'lesion': lesion_criterion,
		'bm': nn.CrossEntropyLoss(),
	}
	weights = {'skin': skin_weight, 'lesion': lesion_weight, 'bm': bm_weight}

	# Optimizer & scheduler (cosine)
	optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
	scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

	# wandb
	if balanced_mixup_cfg['enabled'] and wandb_name is not None and not wandb_name.endswith('_balanced_mixup'):
		wandb_name = f"{wandb_name}_balanced_mixup"
	wandb.init(project=wandb_project, name=wandb_name, config={
		'experiment': experiment,
		'architecture': 'shared_trunk_multi_head',
		'hidden_dims': list(hidden_dims),
		'dropout': dropout,
		'batch_size': batch_size,
		'epochs': epochs,
		'lr': learning_rate,
		'weight_decay': weight_decay,
		'task_weights': weights,
		'lesion_class_weights': lesion_weights.detach().cpu().tolist(),
		'lesion_loss_fn': lesion_loss_fn,
		'focal_gamma': focal_gamma if lesion_loss_fn in ('focal','cb_focal') else None,
		'cb_beta': cb_beta if lesion_loss_fn == 'cb_focal' else None,
		'balanced_mixup': balanced_mixup_cfg,
		'curriculum': curriculum_cfg,
	})
	wandb.watch(model, log='all', log_freq=10)

	# Log class counts and weights for lesion task
	logger.info(f"Lesion class counts: {lesion_weights.detach().cpu().numpy()}")
	logger.info(f"Lesion class weights: {lesion_weights.detach().cpu().tolist()}")
	print(f"Lesion class counts: {lesion_weights.detach().cpu().numpy()}")
	print(f"Lesion class weights: {lesion_weights.detach().cpu().tolist()}")

	best_val_loss = math.inf
	best_train_acc = {'skin': 0.0, 'lesion': 0.0, 'bm': 0.0}
	best_val_acc = {'skin': 0.0, 'lesion': 0.0, 'bm': 0.0}
	for epoch in range(1, epochs+1):
		# Curriculum task weighting
		epoch_idx = epoch - 1
		cur_weights = weights.copy()
		if curriculum_cfg['enabled']:
			t = min(1.0, epoch_idx / max(1, curriculum_cfg['epochs']))
			cur_weights['lesion'] = curriculum_cfg['lesion_start'] + (curriculum_cfg['lesion_end'] - curriculum_cfg['lesion_start']) * t
			cur_weights['bm'] = curriculum_cfg['bm_start'] + (curriculum_cfg['bm_end'] - curriculum_cfg['bm_start']) * t
		wandb.log({'curriculum_lesion_w': cur_weights.get('lesion',1.0), 'curriculum_bm_w': cur_weights.get('bm',1.0), 'epoch': epoch})
		model.train()
		train_losses = []
		train_metrics = {
			'skin': {'preds': [], 'targets': []},
			'lesion': {'preds': [], 'targets': []},
			'bm': {'preds': [], 'targets': []},
		}
		train_correct = {'skin': 0, 'lesion': 0, 'bm': 0}
		train_total = {'skin': 0, 'lesion': 0, 'bm': 0}
		for features, labels, masks, _ in tqdm(train_loader, desc=f'Epoch {epoch} - train'):
			features = features.to(device)
			labels = {k: v.to(device) for k, v in labels.items()}
			masks = {k: v.to(device) for k, v in masks.items()}

			optimizer.zero_grad()
			with torch.amp.autocast('cuda'):
				outputs = model(features)
				# If balanced mixup produces soft lesion labels (float tensor of shape [B, C]),
				# compute cross-entropy with soft targets for lesion; otherwise use standard path.
				if balanced_mixup_cfg['enabled'] and isinstance(labels['lesion'], torch.Tensor) and labels['lesion'].dtype.is_floating_point:
					loss_total = 0.0
					for task in ('skin', 'lesion', 'bm'):
						mask = masks[task]
						valid_idx = mask.nonzero(as_tuple=True)[0]
						if valid_idx.numel() == 0:
							continue
						logits = outputs[task][valid_idx]
						if task == 'lesion':
							targets_soft = labels['lesion'][valid_idx]
							log_probs = torch.log_softmax(logits, dim=1)
							loss_task = -(targets_soft * log_probs).sum(dim=1).mean()
						else:
							targets = labels[task][valid_idx]
							loss_task = criteria[task](logits, targets)
						loss_total = loss_total + weights[task] * loss_task
					loss = loss_total
				else:
					loss = compute_loss(outputs, labels, masks, criteria, cur_weights)
			loss.backward()
			optimizer.step()
			train_losses.append(loss.item())

			# Collect predictions for metrics (only masked examples)
			for task in ('skin', 'lesion', 'bm'):
				mask = masks[task]
				valid_idx = mask.nonzero(as_tuple=True)[0]
				if valid_idx.numel() == 0:
					continue
				pred = outputs[task].argmax(dim=1)[valid_idx].detach().cpu().numpy().tolist()
				# Handle soft labels for lesion during mixup by hardening to argmax for metrics
				if task == 'lesion' and isinstance(labels['lesion'], torch.Tensor) and labels['lesion'].dtype.is_floating_point:
					trg_tensor = labels['lesion'][valid_idx]
					trg = trg_tensor.argmax(dim=1).detach().cpu().numpy().tolist()
				else:
					trg = labels[task][valid_idx].detach().cpu().numpy().tolist()
				train_metrics[task]['preds'].extend(pred)
				train_metrics[task]['targets'].extend(trg)
				train_correct[task] += sum(int(p == t) for p, t in zip(pred, trg))
				train_total[task] += len(trg)

		# Aggregate train metrics
		train_log = {'epoch': epoch, 'train_loss': float(sum(train_losses)/max(1, len(train_losses)))}
		for task in ('skin', 'lesion', 'bm'):
			preds = train_metrics[task]['preds']
			tgts = train_metrics[task]['targets']
			if len(preds) > 0:
				train_log[f'train_{task}_f1'] = f1_score(tgts, preds, average='macro')
				train_log[f'train_{task}_precision'] = precision_score(tgts, preds, average='weighted', zero_division=0)
				train_log[f'train_{task}_recall'] = recall_score(tgts, preds, average='weighted', zero_division=0)
				train_log[f'train_{task}_acc'] = train_correct[task] / max(1, train_total[task])
				best_train_acc[task] = max(best_train_acc[task], train_log[f'train_{task}_acc'])

		# Validation
		model.eval()
		val_losses = []
		val_metrics = {
			'skin': {'preds': [], 'targets': []},
			'lesion': {'preds': [], 'targets': []},
			'bm': {'preds': [], 'targets': []},
		}
		val_correct = {'skin': 0, 'lesion': 0, 'bm': 0}
		val_total = {'skin': 0, 'lesion': 0, 'bm': 0}
		with torch.no_grad():
			for features, labels, masks, _ in tqdm(val_loader, desc=f'Epoch {epoch} - val'):
				features = features.to(device)
				labels = {k: v.to(device) for k, v in labels.items()}
				masks = {k: v.to(device) for k, v in masks.items()}
				with torch.amp.autocast('cuda'):
					outputs = model(features)
					loss = compute_loss(outputs, labels, masks, criteria, cur_weights)
				val_losses.append(loss.item())
				for task in ('skin', 'lesion', 'bm'):
					mask = masks[task]
					valid_idx = mask.nonzero(as_tuple=True)[0]
					if valid_idx.numel() == 0:
						continue
					pred = outputs[task].argmax(dim=1)[valid_idx].detach().cpu().numpy().tolist()
					trg = labels[task][valid_idx].detach().cpu().numpy().tolist()
					val_metrics[task]['preds'].extend(pred)
					val_metrics[task]['targets'].extend(trg)
					val_correct[task] += sum(int(p == t) for p, t in zip(pred, trg))
					val_total[task] += len(trg)

		val_log = {'epoch': epoch}
		if val_losses:
			val_log['val_loss'] = float(sum(val_losses)/len(val_losses))
		for task in ('skin', 'lesion', 'bm'):
			preds = val_metrics[task]['preds']
			tgts = val_metrics[task]['targets']
			if len(preds) > 0:
				val_log[f'val_{task}_f1'] = f1_score(tgts, preds, average='weighted')
				val_log[f'val_{task}_precision'] = precision_score(tgts, preds, average='weighted', zero_division=0)
				val_log[f'val_{task}_recall'] = recall_score(tgts, preds, average='weighted', zero_division=0)
				val_log[f'val_{task}_acc'] = val_correct[task] / max(1, val_total[task])
				best_val_acc[task] = max(best_val_acc[task], val_log[f'val_{task}_acc'])

		# Step cosine scheduler each epoch
		scheduler.step()

		# Log to wandb
		wandb.log({**train_log, **val_log, 'lr': optimizer.param_groups[0]['lr']})

		# Track best
		if 'val_loss' in val_log and val_log['val_loss'] < best_val_loss:
			best_val_loss = val_log['val_loss']
			logger.info(f"New best val loss: {best_val_loss:.4f}")
			
			# Save best checkpoint
			# For exp6 Tier5 runs, save under backend/models/parallel/exp6_tier5_cb_focal
			if experiment == 'exp6':
				if 'exp6_tier5_cb_focal' in wandb_project:
					checkpoint_dir = project_root / 'backend' / 'models' / 'parallel' / 'exp6_tier5_cb_focal' / datetime.now().strftime('%Y%m%d_%H%M%S')
				else:
					checkpoint_dir = project_root / 'backend' / 'models' / 'parallel' / 'exp6_balanced_partial' / datetime.now().strftime('%Y%m%d_%H%M%S')
			else:
				checkpoint_dir = project_root / 'models' / f'{experiment}_parallel' / datetime.now().strftime('%Y%m%d_%H%M%S')
			checkpoint_dir.mkdir(parents=True, exist_ok=True)
			best_checkpoint_path = checkpoint_dir / 'best.pt'
			torch.save({
				'model_state_dict': model.state_dict(),
				'config': {
					'input_dim': 256,
					'hidden_dims': hidden_dims,
					'dropout': dropout,
					'experiment': experiment,
				},
				'epoch': epoch,
				'val_loss': best_val_loss,
			}, best_checkpoint_path)
			logger.info(f"Saved best checkpoint to: {best_checkpoint_path}")

		gc.collect()
		torch.cuda.empty_cache()

	# -----------------------------
	# Test evaluation (always run)
	# -----------------------------
	logger.info("Starting test evaluation...")
	test_dataset = ParallelUnifiedDataset(experiment=experiment, split='test')
	if len(test_dataset) == 0:
		logger.warning("No samples in test split for this experiment. Skipping test evaluation.")
		wandb.finish()
		logger.info("Training completed.")
		return

	test_loader = DataLoader(test_dataset, batch_size=batch_size*2, shuffle=False, num_workers=0, pin_memory=True)
	model.eval()

	test_losses = []
	test_metrics = {
		'skin': {'preds': [], 'targets': []},
		'lesion': {'preds': [], 'targets': []},
		'bm': {'preds': [], 'targets': []},
	}
	test_correct = {'skin': 0, 'lesion': 0, 'bm': 0}
	test_total = {'skin': 0, 'lesion': 0, 'bm': 0}
	with torch.no_grad():
		for features, labels, masks, _ in tqdm(test_loader, desc='Test'):
			features = features.to(device)
			labels = {k: v.to(device) for k, v in labels.items()}
			masks = {k: v.to(device) for k, v in masks.items()}
			with torch.amp.autocast('cuda'):
				outputs = model(features)
				loss = compute_loss(outputs, labels, masks, criteria, weights)
			test_losses.append(loss.item())
			for task in ('skin', 'lesion', 'bm'):
				mask = masks[task]
				valid_idx = mask.nonzero(as_tuple=True)[0]
				if valid_idx.numel() == 0:
					continue
				pred = outputs[task].argmax(dim=1)[valid_idx].detach().cpu().numpy().tolist()
				trg = labels[task][valid_idx].detach().cpu().numpy().tolist()
				test_metrics[task]['preds'].extend(pred)
				test_metrics[task]['targets'].extend(trg)
				test_correct[task] += sum(int(p == t) for p, t in zip(pred, trg))
				test_total[task] += len(trg)

	# Aggregate test metrics
	test_log = {}
	if test_losses:
		test_log['test_loss'] = float(sum(test_losses)/len(test_losses))
	for task in ('skin', 'lesion', 'bm'):
		preds = test_metrics[task]['preds']
		tgts = test_metrics[task]['targets']
		if len(preds) > 0:
			# Accuracy
			acc = (sum(int(p == t) for p, t in zip(preds, tgts)) / len(preds)) if preds else 0.0
			test_log[f'test_{task}_acc'] = acc
			# F1 / Precision / Recall (macro F1)
			test_log[f'test_{task}_f1'] = f1_score(tgts, preds, average='macro')
			test_log[f'test_{task}_precision'] = precision_score(tgts, preds, average='weighted', zero_division=0)
			test_log[f'test_{task}_recall'] = recall_score(tgts, preds, average='weighted', zero_division=0)

	# Log to wandb and print
	if test_log:
		wandb.log(test_log)
		logger.info("Test metrics:")
		for k, v in test_log.items():
			logger.info(f"  {k}: {v:.4f}")

	# Print best epoch-level accuracies for train/val
	logger.info("Best train accuracies (across epochs): " + 
				  ", ".join([f"{k}: {best_train_acc[k]:.4f}" for k in ('skin','lesion','bm')]))
	logger.info("Best val accuracies (across epochs): " + 
				  ", ".join([f"{k}: {best_val_acc[k]:.4f}" for k in ('skin','lesion','bm')]))

	# Save final checkpoint
	if 'checkpoint_dir' in locals():
		final_checkpoint_path = checkpoint_dir / 'final.pt'
		torch.save({
			'model_state_dict': model.state_dict(),
			'config': {
				'input_dim': 256,
				'hidden_dims': hidden_dims,
				'dropout': dropout,
				'experiment': experiment,
			},
			'epoch': epochs,
			'val_loss': val_log.get('val_loss', float('inf')),
		}, final_checkpoint_path)
		logger.info(f"Saved final checkpoint to: {final_checkpoint_path}")

	# Save confusion matrix CSV for lesion head
	try:
		if 'checkpoint_dir' in locals():
			import csv as _csv
			cm = confusion_matrix(test_metrics['lesion']['targets'], test_metrics['lesion']['preds'], labels=list(range(8)))
			cm_csv = checkpoint_dir / 'confusion_matrix_lesion_test.csv'
			with open(cm_csv, 'w', newline='') as f:
				w = _csv.writer(f)
				for row in cm.tolist():
					w.writerow(row)
			logger.info(f"Saved lesion confusion matrix CSV to: {cm_csv}")
	except Exception as _e:
		logger.warning(f"Failed to save confusion matrix CSV: {_e}")

	# Append summary metrics to backend/experiments_summary.csv
	try:
		from pathlib import Path as _P
		import csv as _csv
		backend_csv = project_root / 'backend' / 'experiments_summary.csv'
		backend_csv.parent.mkdir(parents=True, exist_ok=True)
		write_header = not backend_csv.exists()
		with open(backend_csv, 'a', newline='') as f:
			w = _csv.writer(f)
			if write_header:
				w.writerow(['timestamp','experiment','architecture','wandb_project','wandb_name','best_train_skin_acc','best_train_lesion_acc','best_train_bm_acc','best_val_skin_acc','best_val_lesion_acc','best_val_bm_acc','test_skin_acc','test_skin_f1','test_lesion_acc','test_lesion_f1','test_bm_acc','test_bm_f1'])
			w.writerow([
				datetime.now().isoformat(timespec='seconds'),
				experiment,
				'parallel',
				wandb_project,
				wandb_name,
				best_train_acc.get('skin',0.0),
				best_train_acc.get('lesion',0.0),
				best_train_acc.get('bm',0.0),
				best_val_acc.get('skin',0.0),
				best_val_acc.get('lesion',0.0),
				best_val_acc.get('bm',0.0),
				test_log.get('test_skin_acc',0.0),
				test_log.get('test_skin_f1',0.0),
				test_log.get('test_lesion_acc',0.0),
				test_log.get('test_lesion_f1',0.0),
				test_log.get('test_bm_acc',0.0),
				test_log.get('test_bm_f1',0.0),
			])
	except Exception as _e:
		logger.warning(f"Failed to append to backend/experiments_summary.csv: {_e}")

	wandb.finish()
	logger.info("Training completed.")
	
	# Clean up file handler if it was added
	if log_file and 'file_handler' in locals():
		logger.removeHandler(file_handler)
		file_handler.close()
	
	# Clean up file handler if it was added
	if log_file and 'file_handler' in locals():
		logger.removeHandler(file_handler)
		file_handler.close()


if __name__ == '__main__':
	import argparse
	parser = argparse.ArgumentParser(description='Train parallel multi-task model on SAM features with exp4 splits')
	parser.add_argument('--experiment', default='exp4')
	parser.add_argument('--batch_size', type=int, default=32)
	parser.add_argument('--epochs', type=int, default=30)
	parser.add_argument('--lr', type=float, default=1e-3)
	parser.add_argument('--weight_decay', type=float, default=1e-2)
	parser.add_argument('--dropout', type=float, default=0.3)
	parser.add_argument('--hidden_dims', type=str, default='512,256')
	parser.add_argument('--skin_weight', type=float, default=1.0)
	parser.add_argument('--lesion_weight', type=float, default=1.0)
	parser.add_argument('--bm_weight', type=float, default=1.0)
	parser.add_argument('--wandb_project', type=str, default='skin-lesion-classification')
	parser.add_argument('--wandb_name', type=str, default='parallel_exp4')
	args = parser.parse_args()

	hidden_dims = tuple(int(x) for x in args.hidden_dims.split(',') if x)
	train_parallel(
		experiment=args.experiment,
		batch_size=args.batch_size,
		epochs=args.epochs,
		learning_rate=args.lr,
		weight_decay=args.weight_decay,
		dropout=args.dropout,
		hidden_dims=hidden_dims,
		skin_weight=args.skin_weight,
		lesion_weight=args.lesion_weight,
		bm_weight=args.bm_weight,
		wandb_project=args.wandb_project,
		wandb_name=args.wandb_name,
	)


