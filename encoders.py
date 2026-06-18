import torch.nn as nn
import torch
import torch.nn.functional as F 

class NumericEncoder(nn.Module):
	def __init__(self, n_in, hidden=32):
		super().__init__()
		self.net = nn.Sequential(nn.Linear(n_in, hidden), nn.ReLU())
		self.out_dim = hidden
	def forward(self, x): return self.net(x)


class MulticlassEncoder(nn.Module):
	# -- cadinalities: a list with the number of categories for each column
	# -- embed_dim: the amount of columns
	def __init__(self, cardinalities, embed_dim=8):
		super().__init__()
		# -- for each cardinality it trains an embedding
		self.embeds = nn.ModuleList([nn.Embedding(c, embed_dim) for c in cardinalities])
		# -- the amount of output dimensions is directly
		self.out_dim = embed_dim * len(cardinalities)
	def forward(self, x):
		# -- concats 
		return torch.cat([emb(x[:, i]) for i, emb in enumerate(self.embeds)], dim=1)


class BinaryEncoder(nn.Module):
	def __init__(self, n_in, hidden=16):
		super().__init__()
		self.net = nn.Sequential(nn.Linear(n_in, hidden), nn.ReLU())
		self.out_dim = hidden
	def forward(self, x): return self.net(x)


# -- CNN that reduces the image into a vector
class ImageEncoder(nn.Module):
	def __init__(self, in_ch=3, img_size=32, out_dim=64):
		super().__init__()
		self.conv = nn.Sequential(
			nn.Conv2d(in_ch, 16, 3, 2, 1), 	nn.ReLU(), # -- 32x32 -> 16x16
			nn.Conv2d(16, 32, 3, 2, 1), 	nn.ReLU(), # -- 16x16 -> 8x8
			nn.Conv2d(32, 64, 3, 2, 1), 	nn.ReLU(), # -- 8x8 -> 4x4
		)
		# -- the side of the image has been reduce to 4, but in 64 dims
		self.feat_hw = img_size // 8
		# -- creates the final vector
		self.fc = nn.Linear(64*self.feat_hw*self.feat_hw, out_dim)
		self.out_dim = out_dim

	def forward(self, x): return F.relu(self.fc(self.conv(x).flatten(1)))
