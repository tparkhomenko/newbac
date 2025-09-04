#!/usr/bin/env bash
set -euo pipefail

# Configurable env
export WANDB_MODE=online
export WANDB_ENTITY="taisiyaparkhomenko-technische-universit-t-graz"
export WANDB_PROJECT="loss_comparison_weighted_ce_vs_focal_vs_lmf"

EPOCHS=50
BATCH_SIZE=64
LR=1e-4
WEIGHT_DECAY=1e-4
DROPOUT=0.3
HIDDEN_DIMS="512,256"
EXPERIMENT="exp1"

# Ensure output dirs
mkdir -p backend/models/multi backend/models/parallel

# Function to train with a specific loss function
train_with_loss() {
	local loss_fn="$1"
	
	echo "=========================================="
	echo "🚀 Training with lesion_loss_fn: $loss_fn"
	echo "=========================================="
	
	# Update config
	sed -i "s/lesion_loss_fn:.*/lesion_loss_fn: \"$loss_fn\"/" config.yaml
	
	# Verify config update
	echo "Config updated to: $(grep 'lesion_loss_fn:' config.yaml)"
	
	# Run training for both architectures
	for mode in "multi" "parallel"; do
		echo ""
		echo "=== Training $mode architecture with $loss_fn loss ==="
		
		local trainer=""
		local arch_suffix=""
		if [[ "$mode" == "multi" ]]; then
			trainer="training/train_exp5_multihead.py"
			arch_suffix="multihead"
		elif [[ "$mode" == "parallel" ]]; then
			trainer="training/train_parallel.py"
			arch_suffix="parallel"
		fi
		
		local target_dir="backend/models/${mode}"
		local log_file="${target_dir}/logs_${mode}_exp1_8classes_${loss_fn}.txt"
		
		mkdir -p "$target_dir"
		
		# Launch training
		python3 "$trainer" \
			--experiment "$EXPERIMENT" \
			--epochs "$EPOCHS" \
			--batch_size "$BATCH_SIZE" \
			--lr "$LR" \
			--weight_decay "$WEIGHT_DECAY" \
			--dropout "$DROPOUT" \
			--hidden_dims "$HIDDEN_DIMS" \
			--wandb_project "$WANDB_PROJECT" \
			--wandb_name "${EXPERIMENT}_${mode}_8classes_${loss_fn}" \
			2>&1 | tee "$log_file"
		
		# Resolve latest run dir and copy checkpoints
		local models_root="models/${EXPERIMENT}_${arch_suffix}"
		if [[ ! -d "$models_root" ]]; then
			echo "Models dir not found: $models_root" >&2; exit 1
		fi
		local latest_run
		latest_run="$(ls -1dt "${models_root}"/*/ 2>/dev/null | head -n1 || true)"
		if [[ -z "${latest_run:-}" ]]; then
			echo "No run directory under ${models_root}" >&2; exit 1
		fi
		
		# Copy all three MLP checkpoints
		for mlp in mlp1 mlp2 mlp3; do
			local target_ckpt="${target_dir}/${mlp}_${loss_fn}.pt"
			if [[ -f "${latest_run}/best.pt" ]]; then
				cp -f "${latest_run}/best.pt" "$target_ckpt"
				echo "Saved ${mlp}_${loss_fn} checkpoint to ${target_ckpt}"
			elif [[ -f "${latest_run}/final.pt" ]]; then
				cp -f "${latest_run}/final.pt" "$target_ckpt"
				echo "Saved ${mlp}_${loss_fn} checkpoint to ${target_ckpt}"
			else
				echo "No best.pt/final.pt in ${latest_run}" >&2; exit 1
			fi
		done
		
		echo "Logs saved to ${log_file}"
	done
	
	echo ""
	echo "✅ Completed training with $loss_fn loss"
	echo ""
}

# Test all three loss functions
echo "🧪 Testing all lesion loss functions: weighted_ce, focal, lmf"
echo ""

# 1. Weighted CE
train_with_loss "weighted_ce"

# 2. Focal Loss
train_with_loss "focal"

# 3. LMF Loss
train_with_loss "lmf"

echo ""
echo "🎉 All loss function testing completed!"
echo ""
echo "📁 Models saved to:"
echo "  - backend/models/multi/mlp1_*.pt, mlp2_*.pt, mlp3_*.pt"
echo "  - backend/models/parallel/mlp1_*.pt, mlp2_*.pt, mlp3_*.pt"
echo ""
echo "📊 Logs saved to:"
echo "  - backend/models/multi/logs_multi_exp1_8classes_*.txt"
echo "  - backend/models/parallel/logs_parallel_exp1_8classes_*.txt"
echo ""
echo "🔍 WandB Project: $WANDB_PROJECT"
echo "   - 6 runs total (2 architectures × 3 loss functions)"
echo "   - Compare performance across all loss functions"
echo ""
echo "✅ Next steps:"
echo "1. Analyze WandB metrics comparison"
echo "2. Compare final accuracy and F1 scores"
echo "3. Choose best performing loss function for production"
