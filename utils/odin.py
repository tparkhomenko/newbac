import torch
import torch.nn.functional as F
from typing import Tuple


@torch.no_grad()
def _softmax_with_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
	return F.softmax(logits / max(1e-8, float(temperature)), dim=1)


def compute_odin_scores(
	model,
	features: torch.Tensor,
	device: torch.device,
	head: str = 'skin',
	temperature: float = 1000.0,
	epsilon: float = 0.001,
) -> Tuple[torch.Tensor, torch.Tensor]:
	"""
	Compute ODIN scores for a given batch of features on a specified head.

	Returns:
	- ood_scores: Tensor [B] where higher means more likely OOD
	- max_probs: Tensor [B] the max softmax probability after ODIN
	"""
	# Ensure model is in eval mode; grads only for features
	was_training = model.training
	model.eval()

	# First forward to get predicted labels
	features0 = features.detach().to(device)
	with torch.no_grad():
		outputs0 = model(features0)
		logits0 = outputs0[head]
		probs0 = _softmax_with_temperature(logits0, max(1.0, temperature))
		pred_classes = probs0.argmax(dim=1)

	# Prepare features for gradient
	features_req = features.detach().to(device).clone()
	features_req.requires_grad_(True)

	# Forward with temperature and compute loss w.r.t. predicted class
	outputs = model(features_req)
	logits = outputs[head]
	logits_T = logits / max(1e-8, float(temperature))
	# Use NLL of the predicted class
	log_probs = F.log_softmax(logits_T, dim=1)
	loss = -log_probs.gather(1, pred_classes.view(-1, 1)).mean()

	# Backprop to features only: compute gradients first, then freeze model params
	model.zero_grad(set_to_none=True)
	loss.backward()
	
	# Now freeze model parameters after backward pass
	original_requires_grad = []
	for p in model.parameters():
		original_requires_grad.append(p.requires_grad)
		p.requires_grad_(False)

	# Perturbation: sign of gradient
	grad_sign = features_req.grad.detach().sign()
	features_perturbed = features_req.detach() - float(epsilon) * grad_sign

	# Second forward with perturbed input
	with torch.no_grad():
		outputs2 = model(features_perturbed)
		logits2 = outputs2[head]
		probs2 = _softmax_with_temperature(logits2, max(1.0, temperature))
		max_probs = probs2.max(dim=1).values

	# ODIN score: higher score = more OOD
	ood_scores = 1.0 - max_probs

	# Restore model param requires_grad flags
	for p, req in zip(model.parameters(), original_requires_grad):
		p.requires_grad_(req)

	# Restore model training state
	if was_training:
		model.train()

	return ood_scores.detach(), max_probs.detach()



@torch.no_grad()
def compute_msp_scores(
	model,
	features: torch.Tensor,
	device: torch.device,
	head: str = 'skin',
) -> Tuple[torch.Tensor, torch.Tensor]:
	"""
	Compute MSP (Maximum Softmax Probability) OOD scores on a specified head.

	Returns:
	- msp_scores: Tensor [B] where higher means more likely OOD (1 - max_prob)
	- max_probs: Tensor [B] maximum softmax probability per sample
	"""
	was_training = model.training
	model.eval()
	inputs = features.detach().to(device)
	outputs = model(inputs)
	logits = outputs[head]
	probs = F.softmax(logits, dim=1)
	max_probs = probs.max(dim=1).values
	msp_scores = 1.0 - max_probs
	if was_training:
		model.train()
	return msp_scores.detach(), max_probs.detach()


