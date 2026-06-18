import torch.nn as nn
import torch.nn.functional as F 
import torch

class Fusion(nn.Module):
	def __init__(self, in_dm, latent_dim=32, hidden=128):
		super().__init__()
		self.net = nn.Sequential(
			nn.Linear(in_dm, hidden), nn.ReLU(),
			nn.Linear(hidden, latent_dim),
		)
	def forward(self, h): return self.net(h)
