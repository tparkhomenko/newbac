#!/usr/bin/env bash
set -euo pipefail

# Configurable env
export WANDB_MODE=online
export WANDB_ENTITY="taisiyaparkhomenko-technische-universit-t-graz"
export WANDB_PROJECT="uploaded"

EPOCHS=50
BATCH_SIZE=64
LR=1e-4
DROPOUT=0.3
WEIGHT_DECAY=1e-4
HIDDEN_DIMS="512,256"

# Ensure output dirs
mkdir -p backend/models/multi backend/models/parallel

# Map mlp -> experiment
exp_from_mlp() {
	case "$1" in
		mlp1) echo "exp1" ;;
		mlp2) echo "exp2" ;;
		mlp3) echo "exp3" ;;
		*) echo "Unknown mlp: $1" >&2; exit 1 ;;
	 esac
}

# Run trainer for a given mode/mlp
run_one() {
	local mode="$1" mlp="$2"
	local exp; exp="$(exp_from_mlp "$mlp")"

	local trainer=""
	local arch_suffix=""
	if [[ "$mode" == "multi" ]]; then
		trainer="training/train_exp5_multihead.py"
		arch_suffix="multihead"
	elif [[ "$mode" == "parallel" ]]; then
		trainer="training/train_parallel.py"
		arch_suffix="parallel"
	else
		echo "Unknown mode: $mode" >&2; exit 1
	fi

	local target_dir="backend/models/${mode}"
	local target_ckpt="${target_dir}/${mlp}.pt"
	local log_file="${target_dir}/logs_${mlp}.txt"

	echo "=== Training $mode / $mlp (exp=${exp}) ==="
	mkdir -p "$target_dir"

	# Launch training
	python3 "$trainer" \
		--experiment "$exp" \
		--epochs "$EPOCHS" \
		--batch_size "$BATCH_SIZE" \
		--lr "$LR" \
		--weight_decay "$WEIGHT_DECAY" \
		--dropout "$DROPOUT" \
		--hidden_dims "$HIDDEN_DIMS" \
		--wandb_project "$WANDB_PROJECT" \
		--wandb_name "${exp}_${mode}_auto" \
		2>&1 | tee "$log_file"

	# Resolve latest run dir and copy checkpoint
	local models_root="models/${exp}_${arch_suffix}"
	if [[ ! -d "$models_root" ]]; then
		echo "Models dir not found: $models_root" >&2; exit 1
	fi
	local latest_run
	latest_run="$(ls -1dt "${models_root}"/*/ 2>/dev/null | head -n1 || true)"
	if [[ -z "${latest_run:-}" ]]; then
		echo "No run directory under ${models_root}" >&2; exit 1
	fi

	if [[ -f "${latest_run}/best.pt" ]]; then
		cp -f "${latest_run}/best.pt" "$target_ckpt"
	elif [[ -f "${latest_run}/final.pt" ]]; then
		cp -f "${latest_run}/final.pt" "$target_ckpt"
	else
		echo "No best.pt/final.pt in ${latest_run}" >&2; exit 1
	fi

	echo "Saved checkpoint to ${target_ckpt}"
	echo "Logs saved to ${log_file}"
}

modes=("multi" "parallel")
mlps=("mlp1" "mlp2" "mlp3")

for mode in "${modes[@]}"; do
	for mlp in "${mlps[@]}"; do
		run_one "$mode" "$mlp"
	done
done

echo "All runs completed."


