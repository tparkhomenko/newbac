#!/usr/bin/env python3
import os
import sys
import json
import csv
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from datasets.parallel_unified_dataset import ParallelUnifiedDataset
from models.multitask_model import MultiTaskHead
from utils.lesion_mapping import LESION_CLASS_MAPPING, lesion_requires_bm


def load_model(ckpt_path: Path, device: torch.device) -> torch.nn.Module:
	ckpt = torch.load(ckpt_path, map_location=device)
	model = MultiTaskHead(
		input_dim=256,
		hidden_dims=(512, 256),
		dropout=0.3,
		num_classes_skin=2,
		num_classes_lesion=8,
		num_classes_bm=2,
		num_classes_final=11,  # Add final head for exp_finalmulticlass
	).to(device)
	model.load_state_dict(ckpt['model_state_dict'])
	model.eval()
	return model


def evaluate_headwise(model: torch.nn.Module, dataloader, device: torch.device) -> Dict[str, float]:
	skin_preds, skin_tgts = [], []
	lesion_preds, lesion_tgts = [], []
	bm_preds, bm_tgts = [], []
	with torch.no_grad():
		for features, labels, masks, _ in tqdm(dataloader, desc='Headwise eval'):
			features = features.to(device)
			labels = {k: v.to(device) for k, v in labels.items()}
			masks = {k: v.to(device) for k, v in masks.items()}
			outputs = model(features)
			for task in ('skin','lesion','bm'):
				valid_idx = masks[task].nonzero(as_tuple=True)[0]
				if valid_idx.numel() == 0: continue
				logits = outputs[task][valid_idx]
				pred = logits.argmax(dim=1)
				trg = labels[task][valid_idx]
				if task == 'skin':
					skin_preds.extend(pred.cpu().numpy().tolist())
					skin_tgts.extend(trg.cpu().numpy().tolist())
				elif task == 'lesion':
					lesion_preds.extend(pred.cpu().numpy().tolist())
					lesion_tgts.extend(trg.cpu().numpy().tolist())
				else:
					bm_preds.extend(pred.cpu().numpy().tolist())
					bm_tgts.extend(trg.cpu().numpy().tolist())
	metrics = {}
	if skin_preds:
		metrics['skin_acc'] = accuracy_score(skin_tgts, skin_preds)
		metrics['skin_f1'] = f1_score(skin_tgts, skin_preds, average='macro')
	if lesion_preds:
		metrics['lesion_acc'] = accuracy_score(lesion_tgts, lesion_preds)
		metrics['lesion_f1'] = f1_score(lesion_tgts, lesion_preds, average='macro')
	if bm_preds:
		metrics['bm_acc'] = accuracy_score(bm_tgts, bm_preds)
		metrics['bm_f1'] = f1_score(bm_tgts, bm_preds, average='macro')
	return metrics


def evaluate_task_aware(model: torch.nn.Module, dataloader, device: torch.device, lesion_map: Dict[str,str]) -> Tuple[Dict[str, float], np.ndarray, np.ndarray, Dict[str,int]]:
	# You may adapt mapping to your lesion index-to-name mapping; here we assume an index->name mapping
	idx_to_lesion = {0:'melanoma',1:'nevus',2:'bcc',3:'scc',4:'seborrheic_keratosis',5:'akiec',6:'lentigo',7:'vasc'}
	final_preds = []
	final_tgts = []
	label_to_idx: Dict[str,int] = {}
	with torch.no_grad():
		for features, labels, masks, _ in tqdm(dataloader, desc='Task-aware eval'):
			features = features.to(device)
			labels = {k: v.to(device) for k, v in labels.items()}
			masks = {k: v.to(device) for k, v in masks.items()}
			outputs = model(features)
			skin_logits = outputs['skin']
			lesion_logits = outputs['lesion']
			bm_logits = outputs['bm']
			B = features.shape[0]
			for i in range(B):
				# GT label aggregation for pipeline-level (string-based)
				# We build a best-effort GT label consistent with decision flow for confusion matrix
				# When skin mask is invalid, skip sample in pipeline metrics
				if masks['skin'][i].item() == 0: 
					continue
				skin_pred = skin_logits[i].argmax().item()
				if skin_pred == 0:
					pred_label = 'not_skin'
				else:
					lesion_idx = lesion_logits[i].argmax().item()
					lesion_name = idx_to_lesion.get(lesion_idx, 'unknown')
					if lesion_mapping_requires_bm(lesion_name, lesion_map):
						if masks['bm'][i].item() == 0:
							# cannot decide BM; fallback benign
							bm_name = 'benign'
						else:
							bm_pred = bm_logits[i].argmax().item()
							bm_name = 'malignant' if bm_pred == 1 else 'benign'
						pred_label = f"{bm_name}_{lesion_name}"
					else:
						pred_label = f"benign_{lesion_name}"
				# Build GT label similarly (approximate)
				gt_skin = labels['skin'][i].item()
				if gt_skin == 0:
					gt_label = 'not_skin'
				else:
					if masks['lesion'][i].item() == 1:
						gt_lesion_idx = labels['lesion'][i].item()
						gt_lesion_name = idx_to_lesion.get(int(gt_lesion_idx), 'unknown')
						if lesion_mapping_requires_bm(gt_lesion_name, lesion_map) and masks['bm'][i].item() == 1:
							bm_val = labels['bm'][i].item()
							bm_name = 'malignant' if bm_val == 1 else 'benign'
							gt_label = f"{bm_name}_{gt_lesion_name}"
						else:
							gt_label = f"benign_{gt_lesion_name}"
					else:
						gt_label = 'unknown'
				# map to indices
				for lab in (pred_label, gt_label):
					if lab not in label_to_idx:
						label_to_idx[lab] = len(label_to_idx)
				final_preds.append(label_to_idx[pred_label])
				final_tgts.append(label_to_idx[gt_label])
	# metrics
	final_preds = np.array(final_preds, dtype=int)
	final_tgts = np.array(final_tgts, dtype=int)
	acc = float(accuracy_score(final_tgts, final_preds)) if len(final_tgts) else 0.0
	f1 = float(f1_score(final_tgts, final_preds, average='macro')) if len(final_tgts) else 0.0
	cm = confusion_matrix(final_tgts, final_preds, labels=list(range(len(label_to_idx)))) if len(final_tgts) else np.zeros((1,1), dtype=int)
	return {'pipeline_acc': acc, 'pipeline_f1': f1}, cm, final_tgts, label_to_idx


def lesion_mapping_requires_bm(lesion: str, override_map: Dict[str,str]) -> bool:
	if override_map:
		return (override_map.get(lesion, 'bm') == 'bm')
	return lesion_requires_bm(lesion)


def save_confusion_csv_png(cm: np.ndarray, label_to_idx: Dict[str,int], out_dir: Path, prefix: str) -> None:
	labels = [k for k,_ in sorted(label_to_idx.items(), key=lambda kv: kv[1])]
	# CSV
	with open(out_dir / f'{prefix}_confusion.csv', 'w', newline='') as f:
		w = csv.writer(f)
		w.writerow([''] + labels)
		for i, row in enumerate(cm.tolist()):
			w.writerow([labels[i]] + row)
	# PNG
	try:
		import seaborn as sns
		plt.figure(figsize=(max(8, len(labels)*0.6), max(6, len(labels)*0.6)))
		import pandas as pd
		df = pd.DataFrame(cm, index=labels, columns=labels)
		sns.heatmap(df, annot=False, cmap='Blues', cbar=True)
		plt.title(f'{prefix} Confusion Matrix')
		plt.ylabel('GT'); plt.xlabel('Pred')
		plt.tight_layout()
		plt.savefig(out_dir / f'{prefix}_confusion.png', dpi=200, bbox_inches='tight')
		plt.close()
	except Exception:
		pass


def main() -> None:
	import argparse
	ap = argparse.ArgumentParser()
	ap.add_argument('--checkpoint', required=True)
	ap.add_argument('--run_name', required=True)
	ap.add_argument('--wandb_project', default='mlp_taskaware_eval')
	ap.add_argument('--eval_mode', choices=['headwise','taskaware','both'], default='both')
	ap.add_argument('--lesion_mapping', type=str, default=None, help='Path to JSON mapping dict')
	args = ap.parse_args()

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	model = load_model(Path(args.checkpoint), device)
	ds = ParallelUnifiedDataset(experiment='exp1', split='test')
	dl = torch.utils.data.DataLoader(ds, batch_size=128, shuffle=False)

	# Load override mapping if any
	override = None
	if args.lesion_mapping:
		with open(args.lesion_mapping, 'r') as f:
			override = json.load(f)

	results = {}
	if args.eval_mode in ('headwise','both'):
		results['headwise'] = evaluate_headwise(model, dl, device)
		print('Headwise metrics:', results['headwise'])
	if args.eval_mode in ('taskaware','both'):
		pipeline_metrics, cm, tgts, label_to_idx = evaluate_task_aware(model, dl, device, override or LESION_CLASS_MAPPING)
		results['taskaware'] = pipeline_metrics
		print('Task-aware metrics:', pipeline_metrics)
		# Save under checkpoint dir
		ckpt_dir = Path(args.checkpoint).parent
		save_confusion_csv_png(cm, label_to_idx, ckpt_dir, prefix='taskaware')

	# Log to W&B
	try:
		import wandb
		wandb.init(project=args.wandb_project, name=args.run_name, tags=['headwise_eval' if args.eval_mode in ('headwise','both') else None, 'taskaware_eval' if args.eval_mode in ('taskaware','both') else None])
		wandb.log(results.get('headwise', {}))
		wandb.log(results.get('taskaware', {}))
		wandb.finish()
	except Exception as e:
		print(f"[WARN] W&B logging failed: {e}")

	# Append to backend/experiments_summary.csv
	out_csv = ROOT / 'backend' / 'experiments_summary.csv'
	head = ['timestamp','run_name','checkpoint','mode','skin_acc','skin_f1','lesion_acc','lesion_f1','bm_acc','bm_f1','pipeline_acc','pipeline_f1']
	import datetime
	row = [datetime.datetime.now().isoformat(timespec='seconds'), args.run_name, str(Path(args.checkpoint)), args.eval_mode]
	row += [results.get('headwise',{}).get('skin_acc'),results.get('headwise',{}).get('skin_f1'),results.get('headwise',{}).get('lesion_acc'),results.get('headwise',{}).get('lesion_f1'),results.get('headwise',{}).get('bm_acc'),results.get('headwise',{}).get('bm_f1')]
	row += [results.get('taskaware',{}).get('pipeline_acc'),results.get('taskaware',{}).get('pipeline_f1')]
	write_header = not out_csv.exists()
	with open(out_csv, 'a', newline='') as f:
		w = csv.writer(f)
		if write_header:
			w.writerow(head)
		w.writerow(row)


if __name__ == '__main__':
	main()
