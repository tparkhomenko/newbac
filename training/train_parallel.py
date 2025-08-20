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
from sklearn.metrics import f1_score, precision_score, recall_score
from tqdm import tqdm
import wandb

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from datasets.parallel_unified_dataset import ParallelUnifiedDataset
from models.multitask_model import MultiTaskHead


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
		wandb_name: str = 'parallel_exp4'
):
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	logger.info(f"Using device: {device}")

	# Datasets and loaders
	train_dataset = ParallelUnifiedDataset(experiment=experiment, split='train')
	val_dataset = ParallelUnifiedDataset(experiment=experiment, split='val')

	train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
	val_loader = DataLoader(val_dataset, batch_size=batch_size*2, shuffle=False, num_workers=0, pin_memory=True)

	# Model
	model = MultiTaskHead(input_dim=256, hidden_dims=tuple(hidden_dims), dropout=dropout)
	model = model.to(device)

	# Losses
	criteria = {
		'skin': nn.CrossEntropyLoss(),
		'lesion': nn.CrossEntropyLoss(),
		'bm': nn.CrossEntropyLoss(),
	}
	weights = {'skin': skin_weight, 'lesion': lesion_weight, 'bm': bm_weight}

	# Optimizer & scheduler
	optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
	scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

	# wandb
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
	})
	wandb.watch(model, log='all', log_freq=10)

	best_val_loss = math.inf
	best_train_acc = {'skin': 0.0, 'lesion': 0.0, 'bm': 0.0}
	best_val_acc = {'skin': 0.0, 'lesion': 0.0, 'bm': 0.0}
	for epoch in range(1, epochs+1):
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
				loss = compute_loss(outputs, labels, masks, criteria, weights)
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
				train_log[f'train_{task}_f1'] = f1_score(tgts, preds, average='weighted')
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
					loss = compute_loss(outputs, labels, masks, criteria, weights)
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

		# Step scheduler on validation loss
		if 'val_loss' in val_log:
			scheduler.step(val_log['val_loss'])

		# Log to wandb
		wandb.log({**train_log, **val_log, 'lr': optimizer.param_groups[0]['lr']})

		# Track best
		if 'val_loss' in val_log and val_log['val_loss'] < best_val_loss:
			best_val_loss = val_log['val_loss']
			logger.info(f"New best val loss: {best_val_loss:.4f}")
			
			# Save best checkpoint
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
			# F1 / Precision / Recall
			test_log[f'test_{task}_f1'] = f1_score(tgts, preds, average='weighted')
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

	wandb.finish()
	logger.info("Training completed.")


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


