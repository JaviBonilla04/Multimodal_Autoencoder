import torch
from torch.utils.data import Dataset

import os
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Dict
from PIL import Image

# -- 	standarized al modalities:
#		numeric_cols 	-> mean: 0, desv: 1
#		multiclass_cols -> turned into an index: 0,...,c-1
#		binary_cols		-> float representation: 0.0, 1.0
class Preprocessor:
	def __init__(self, numeric_cols: List[str], multiclass_cols: List[str], binary_cols: List[str]):
		self.numeric_cols		= list(numeric_cols)
		self.multiclass_cols 	= list(multiclass_cols)
		self.binary_cols 		= list(binary_cols)	

		# -- the hyperparamenter for numeric estandarization
		self.num_mean	: np.ndarray | None = None
		self.num_std	: np.ndarray | None = None

		# -- dict of dictionaries that map the categorical multiclass columns (label to index)
		self.mc_vocab: Dict[str, Dict] = {}
		# -- to map from 0 to 0.0, and 1 to 1.0
		self.bin_map: Dict[str, Dict] = {}

		self.num_fill : Dict[str, float] = {}

	def fit(self, df: pd.DataFrame):
		# -- numeric data
		if self.numeric_cols:
			for c in self.numeric_cols:
				self.num_fill[c] = float(df[c].median())
			arr = df[self.numeric_cols].fillna(self.num_fill).to_numpy(dtype=np.float32)
			self.num_mean = arr.mean(axis=0)
			self.num_std = arr.std(axis=0)
			self.num_std[self.num_std == 0] = 1.0

			# -- make the standard deviation equal to 1 in the case of havin a constant row
			self.num_std[self.num_std == 0] = 1.0

		# -- multiclass data -> build the vocab per multiclass
		for col in self.multiclass_cols:
			serie =  df[col].fillna("__missing__")
			cats = sorted(serie.unique().tolist(), key=lambda v : str(v))
			# -- assinfs a consecutive index per category
			self.mc_vocab[col] = {val: i for i, val in enumerate(cats)}

		# -- binary data
		for col in self.binary_cols:
			serie = df[col].fillna("__missing__")
			# -- to ensure that the cols contains only to possible values: 0 and 1
			#vals = sorted(df[col].dropna().unique().tolist(), key=lambda v: str(v))
			vals = sorted(serie.unique().tolist(), key=lambda v: str(v))
			self.bin_map[col] = {v: float(i) for i, v in enumerate(vals)}
			"""
			if len(vals) > 2:
				raise ValueError(f"binary column col {col} contains >2 values")
			# -- converts from the identifier alphabetically into 0.0 and 1.0
			# -- this works for 0/1, 0.0/1.0, true/false, si/no, yes/no
			self.bin_map[col] = {v: float(i) for i, v in enumerate(vals)}
			"""
		# -- to access fit outside
		return self

	def transform_row(self, row: pd.Series):
		# -- apply numeric correction mean = 0 and desv = 1
		if self.numeric_cols:
			num = np.array([row[c] if pd.notna(row[c]) else self.num_fill[c] for c in self.numeric_cols], dtype=np.float32)
			#num = np.array([row[c] for c in self.numeric_cols], dtype=np.float32)
			num = (num - self.num_mean) / self.num_std
		else: 
			num = np.zeros(0, dtype=np.float32)

		# -- apply multiclass dictionaries
		mc = np.array(
			[self.mc_vocab[c].get(row[c] if pd.notna(row[c]) else "__missing__", 0) for c in self.multiclass_cols],
			dtype=np.int64,
		)

		# -- apply binary mapping to 0.0 and 1.0
		bn = np.array(
			[self.bin_map[c].get(row[c] if pd.notna(row[c]) else "__missing__", 0.0) for c in self.binary_cols],
			dtype=np.float32,
		)

		return num, mc, bn

	# -- the amount of classes that the different multiclasses have
	@property
	def multiclass_cardinalities(self) -> List[int]:
		return [len(self.mc_vocab[c]) for c in self.multiclass_cols]

class MultimodalCSVDataset(Dataset):
	def __init__(self, df: pd.DataFrame, pre: Preprocessor, 
				image_dir: str, image_col: str,
				img_size: int = 32, img_channels: int = 3, resize_mode: str = "center_crop"):
		self.df 			= df.reset_index(drop=True) # -- reset the index
		self.pre 			= pre
		self.image_dir 		= image_dir
		self.image_col 		= image_col
		self.image_size 	= img_size
		self.img_channels 	= img_channels				# -- expects RGB images
		self.resize_mode	= resize_mode

	# -- amount of rows
	def __len__(self):
		return len(self.df)

	def _load_image(self, filename: str) -> torch.Tensor:
		path = os.path.join(self.image_dir, str(filename))
		mode = "RGB" if self.img_channels == 3 else "L"
		with Image.open(path) as im:
			im = im.convert(mode)

			if self.resize_mode == "stretch":
				im = im.resize((self.image_size, self.image_size))
			elif self.resize_mode == "center_crop":
				w, h = im.size
				scale = self.image_size / min(w, h)
				new_w, new_h = round(w*scale), round(h*scale)
				im = im.resize((new_w, new_h))
				left = (new_w - self.image_size) // 2
				top = (new_h - self.image_size) // 2
				im = im.crop((left, top, left + self.image_size, top + self.image_size))
			else:
				raise ValueError(f"resize mode unknown: {self.resize_mode}")

			arr = np.asarray(im, dtype=np.float32) / 255.0 # -- the model awaits pixel range [0, 1]
		if self.img_channels == 1:
			arr = arr[None, :, :]
		else:
			arr = np.transpose(arr, (2, 0, 1))
		return torch.from_numpy(arr)

	def __getitem__(self, idx: int):
		row = self.df.iloc[idx]
		num, mc, bn = self.pre.transform_row(row)
		image = self._load_image(row[self.image_col])
		return {
			"numeric"		: torch.from_numpy(num),
			"multiclass"	: torch.from_numpy(mc),
			"binary"		: torch.from_numpy(bn),
			"image"			: image,
		}


# -- handler of the model dimensions
@dataclass
class Schema:
	n_numeric					: int
	multiclass_cardinalities	: List[int]
	n_binary					: int
	img_channels				: int
	image_size					: int

def build_schema(pre: Preprocessor, img_channels: int, img_size: int) -> Schema:
	return Schema(
		n_numeric 					= len(pre.numeric_cols),
		multiclass_cardinalities 	= pre.multiclass_cardinalities,
		n_binary 					= len(pre.binary_cols),
		img_channels 				= img_channels,
		image_size 					= img_size,
	)

class MultimodalDataset(Dataset):
	def __init__(self, numeric, multiclass, binary, images):
		# numeric:			tensor float
		# multiclass:		tensor long
		# binary:			tensor float
		# images:			tensor float
		self.numeric 	= 	numeric
		self.multiclass = 	multiclass
		self.binary 	= 	binary
		self.images 	= 	images

	# -- Returns the number of rows
	def __len__(self):
		return len(self.numeric)

	# -- Returns the row in the index 
	def __getitem__(self, i):
		return {
			"numeric":		self.numeric[i],
			"multiclass":	self.multiclass[i],
			"binary":		self.binary[i],
			"image":		self.images[i],
		}

	def samples(self, N=5):
		for i in range(N):
			item = self.__getitem__(i)
			numeric = item['numeric']
			multiclass = item['multiclass']
			binary = item['binary']
			image = item['image']

			print(f"numeric: {numeric}")
			print(f"multiclass: {multiclass}")
			print(f"binary: {binary}")
			print(f"image: {image}")
			print("\n")


# -- testing by creating a random dataset
def test_dataset_handler():
	N 				= 512
	numeric 		= torch.randn(N, 8)
	multiclass 		= torch.stack([torch.randint(0, c, (N, )) for c in [5, 4, 10]], dim=1)
	binary			= torch.randint(0, 2, (N, 6)).float()
	images			= torch.rand(N, 3, 32, 32)  

	dataset = MultimodalDataset(
		numeric 	= numeric,
		multiclass 	= multiclass,
		binary 		= binary,
		images 		= images
	)

	dataset.samples(5)

def main():
	pass

if __name__ == "__main__":
	main()


