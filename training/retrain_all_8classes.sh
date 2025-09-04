#!/usr/bin/env bash
set -euo pipefail

# Configurable env
export WANDB_MODE=online
export WANDB_ENTITY="taisiyaparkhomenko-technische-universit-t-graz"
export WANDB_PROJECT="uploaded"

EPOCHS=50
BATCH_SIZE=64
LR=1e-4
WEIGHT_DECAY=1e-4
DROPOUT=0.3
HIDDEN_DIMS="512,256"

# Ensure output dirs
mkdir -p backend/models/multi backend/models/parallel

# MixUp suffix from config
BM_ENABLED=false
if [[ -f "config.yaml" ]]; then
	BM_FLAG=$(awk '/^balanced_mixup:/{f=1;next} f && /enabled:/{print $2; exit}' config.yaml | tr 'A-Z' 'a-z')
	if [[ "${BM_FLAG}" == "true" ]]; then BM_ENABLED=true; fi
fi
SUFFIX=""
if ${BM_ENABLED}; then
	echo "Balanced MixUp enabled → suffix added"
	SUFFIX="_balanced_mixup"
else
	echo "Balanced MixUp disabled → no suffix"
fi

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
	local logs_dir="${target_dir}/logs"
	mkdir -p "$logs_dir"
	local target_ckpt="${target_dir}/${mlp}${SUFFIX}.pt"
	local log_file="${logs_dir}/logs_${mlp}_8classes${SUFFIX}.txt"

	echo "=== Training $mode / $mlp (exp=${exp}) with 8 fine-grained lesion classes ==="
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
		--wandb_name "${exp}_${mode}_8classes${SUFFIX}" \
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

echo "All runs completed with 8 fine-grained lesion classes!"
echo ""
echo "Summary of changes made:"
echo "1. ✅ Dataset updated: 8 fine-grained lesion classes (MEL, NV, BCC, SCC, BKL, AKIEC, DF, VASC)"
echo "2. ✅ NOT_SKIN handled exclusively by MLP1 (skin vs not-skin classifier)"
echo "3. ✅ Config updated: mlp2.output_dim = 8, class_names updated"
echo "4. ✅ Model updated: MultiTaskHead now uses num_classes_lesion=8"
echo "5. ✅ Backend updated: inference pipeline handles 8 classes"
echo "6. ✅ All models retrained: both multihead and parallel architectures"
echo ""
echo "Models saved to:"
echo "  - backend/models/multi/mlp1.pt, mlp2.pt, mlp3.pt"
echo "  - backend/models/parallel/mlp1.pt, mlp2.pt, mlp3.pt"
echo ""
echo "Next steps:"
echo "1. Test the new models with Quick Test"
echo "2. Verify confusion matrix shows 8x8 grid for lesion classification"
echo "3. Check frontend displays all 8 lesion types correctly"
echo "4. Verify MLP1 correctly handles skin vs not-skin separately"
