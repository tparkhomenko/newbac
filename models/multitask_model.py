import torch
import torch.nn as nn
from typing import Dict, Tuple


class MultiTaskHead(nn.Module):

	def __init__(self, input_dim: int, hidden_dims: Tuple[int, ...], dropout: float,
	             num_classes_skin: int = 2, num_classes_lesion: int = 5, num_classes_bm: int = 2):
		super().__init__()
		layers = []
		current_dim = input_dim
		for hidden_dim in hidden_dims:
			layers.append(nn.Linear(current_dim, hidden_dim))
			layers.append(nn.ReLU())
			layers.append(nn.Dropout(dropout))
			current_dim = hidden_dim
		self.trunk = nn.Sequential(*layers) if layers else nn.Identity()
		self.head_skin = nn.Linear(current_dim, num_classes_skin)
		self.head_lesion = nn.Linear(current_dim, num_classes_lesion)
		self.head_bm = nn.Linear(current_dim, num_classes_bm)

	def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
		shared = self.trunk(x)
		return {
			'skin': self.head_skin(shared),
			'lesion': self.head_lesion(shared),
			'bm': self.head_bm(shared)
		}


