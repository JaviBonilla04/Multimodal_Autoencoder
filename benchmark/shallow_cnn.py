import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from PIL import Image

CSV_PATH 	= "C:\\Users\\JB\\Desktop\\ML_Proyectos\\multimodal_learning\\autoencoder\\datasets\\metadata.csv"
IMAGE_DIR 	= "C:\\Users\\JB\\Desktop\\ML_Proyectos\\multimodal_learning\\autoencoder\\datasets\\imgs_part_1\\imgs_part_1"


IMAGE_COL 	= "img_id"
IMG_SIZE 	= 32

BATCH 		= 32
EPOCHS 		= 50
LR 			= 1e-3
CANCER 		= {"BCC", "MEL", "SCC"} # -- the categories on the diagnostic column that represent a type of cancer

class ImageOnlyDataset(Dataset):
	def __init__(self, df, image_dir, image_col, img_size):
		self.df 		= df.reset_index(drop=True)
		self.image_dir 	= image_dir
		self.image_col 	= image_col
		self.image_size = img_size
		self.labels 	= self.df["diagnostic"].isin(CANCER).astype(np.float32).values

	def __len__(self):
		return len(self.df)

	def _load(self, fn):
		path = os.path.join(self.image_dir, str(fn))
		with Image.open(path) as im:
			im 		= im.convert("RGB")
			w, h 	= im.size
			s 		= self.image_size / min(w, h)
			im 		= im.resize((round(w*s), round(h*s)))
			l 		= (im.size[0] - self.image_size) // 2
			t 		= (im.size[1] - self.image_size) // 2
			im 		= im.crop((l, t, l+self.image_size, t+self.image_size))
			arr 	= np.asarray(im, dtype=np.float32) / 255.0
		return torch.from_numpy(np.transpose(arr, (2, 0, 1)))

	def __getitem__(self, i):
		row = self.df.iloc[i]
		return self._load(row[self.image_col]), torch.tensor(self.labels[i])

class ShallowCNN(nn.Module):
	def __init__(self, in_ch=3, img_size=32, out_dim=1):
		super().__init__()
		self.conv = nn.Sequential(
			nn.Conv2d(in_ch, 16, 3, 2, 1), nn.ReLU(),
			nn.Conv2d(16, 32, 3, 2, 1), nn.ReLU(),
			nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),
		)

		self.feat_hw = img_size // 8
		self.fc = nn.Linear(64*self.feat_hw*self.feat_hw, out_dim)
		self.out_dim = out_dim

	def forward(self, x): return F.relu(self.fc(self.conv(x).flatten(1)))


def eval(logits, y_true, umbral=0.5):
	probs	= torch.sigmoid(logits)
	y_preds	= (probs >= umbral).float()
	y_true	= y_true.float()

	tp = ((y_pred == 1) & (y_true == 1)).sum().item()
	tn = ((y_pred == 0) & (y_true == 0)).sum().item()
	fp = ((y_pred == 1) & (y_true == 0)).sum().item()
	fn = ((y_pred == 0) & (y_true == 1)).sum().item()

	acc 	= (tp+tn)/max(tp+tn+fp+fn, 1)
	prec 	= tp/max(tp+fp, 1)
	rec 	= tp/max(tp+fn, 1)
	f1 		= 2*prec*rec/max(prec+rec, 1e-9)

	return {
		"acc" 	: acc,
		"prec"	: prec,
		"rec" 	: rec,
		"f1" 	: f1,
		"tp" 	: tp,
		"tn" 	: tn,
		"fp" 	: fp,
		"fn" 	: fn,
	}



def graph_loss_schema(hist, title="Loss By Modality", save=False, scale_log=False):
 	plt.figure(figsize=(9, 5))

 	if isinstance(hist, dict):
 		curves = hist
 	else:
 		curvers = {"loss": hist}

 	for name, vals in curves.items():
 		epoch = range(1, len(vals)+1)
 		plt.plot(epoch, vals, marker="o", markersize=3, label=name)


 	plt.xlabel("Epoch")
 	plt.ylabel("Loss")
 	plt.title(title)
 	
 	if scale_log:
 		plt.yscale("log")
 	plt.grid(True, alpha=0.3)
 	if len(curves) > 1:
 		plt.legend()
 	plt.tight_layout()

 	if save:
 		plt.savefig(save, dpi=150)
 		print(f"Saved graph in {save}")
 	plt.show()



def main():
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	df = pd.read_csv(CSV_PATH)
	df = df.dropna(subset=["diagnostic", IMAGE_COL]).reset_index(drop=True)

	train_df, val_df = train_test_split(df, test_size=0.2, random_state=0)

	train_ds = ImageOnlyDataset(train_df, IMAGE_DIR, IMAGE_COL, IMG_SIZE)
	val_ds = ImageOnlyDataset(val_df, IMAGE_DIR, IMAGE_COL, IMG_SIZE)

	print(f"CANCER: {CANCER}")

	train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
	val_loader = DataLoader(val_ds, batch_size=BATCH, shuffle=False)

	y_tr = train_ds.labels
	n_pos = y_tr.sum()
	n_neg = len(y_tr) - n_pos

	pos_weight = torch.tensor([n_neg / max(n_pos, 1)], device=device)
	print(f"pos_weight = {pos_weight.item():.2f}")

	model = ShallowCNN().to(device)
	opt = torch.optim.Adam(model.parameters(), lr=LR)
	loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

	loss_class = []

	for epoch in range(1, EPOCHS+1):
		model.train()
		tot = 0.0

		for x, y in train_loader:
			x, y = x.to(device), y.to(device)
			opt.zero_grad()
			logits = model(x).squeeze(1)
			loss = loss_fn(logits, y)
			loss.backward()
			opt.step()

			loss_class.append(float(loss.item()))

			tot += loss.item()
		if epoch % 1 == 0:
			print(f"	epoch {epoch:3d} | loss {tot/len(train_loader):.4f}")

	model.eval()
	all_logits, all_y = [], []

	with torch.no_grad():
		for x, y in val_loader:
			x = x.to(device)
			all_logits.append(model(x).squeeze(1).cpu())
			all_y.append(y)
	logits = torch.cat(all_logits)
	y_va = torch.cat(all_y)
	m = eval(logits, y_va)

	print("MobileNetV3_small (desde cero, solo imagen) — VAL")
	print("=" * 50)
	print(f"  Accuracy : {m['acc']:.3f}")
	print(f"  Precision: {m['prec']:.3f}")
	print(f"  Recall   : {m['rec']:.3f}")
	print(f"  F1       : {m['f1']:.3f}")
	print(f"\n  Matriz de confusión:")
	print(f"                 pred neg   pred pos")
	print(f"    real neg  |    {m['tn']:5d}      {m['fp']:5d}")
	print(f"    real pos  |    {m['fn']:5d}      {m['tp']:5d}")

	graph_loss_schema(loss_class, "Shallow CNN Loss")

if __name__=="__main__":
	main()