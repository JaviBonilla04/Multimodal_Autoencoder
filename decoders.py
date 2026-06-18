import torch.nn as nn
import torch.nn.functional as F 
import torch

class NumericDecoder(nn.Module):
	def __init__(self, latent, n_out, hidden=32):
		super().__init__()
		self.net = nn.Sequential(
			nn.Linear(latent, hidden), nn.ReLU(),
			nn.Linear(hidden, n_out)
		)
	def forward(self, z): return self.net(z)


class MulticlassDecoder(nn.Module):
	def __init__(self, latent, cardinalities):
		super().__init__()
		self.heads = nn.ModuleList([nn.Linear(latent, c) for c in cardinalities])
	def forward(self, z): return [h(z) for h in self.heads]


class BinaryDecoder(nn.Module):
	def __init__(self, latent, n_out):
		super().__init__()
		self.net = nn.Linear(latent, n_out)
	def forward(self, z): return self.net(z)


class ImageDecoder(nn.Module):
	def __init__(self, latent, out_ch=3, feat_hw=4):
		super().__init__()
		self.feat_hw = feat_hw
		self.fc = nn.Linear(latent, 64*feat_hw*feat_hw)
		self.deconv = nn.Sequential(
			nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),  # 4  -> 8
            nn.ConvTranspose2d(32, 16, 4, 2, 1), nn.ReLU(),  # 8  -> 16
            nn.ConvTranspose2d(16, out_ch, 4, 2, 1), nn.Sigmoid(),  # 16 -> 32
		)
	def forward(self, z):
		h = self.fc(z).view(-1, 64, self.feat_hw, self.feat_hw)
		return self.deconv(h)
