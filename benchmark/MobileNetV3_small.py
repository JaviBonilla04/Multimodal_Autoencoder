import os
import numpy as np
import torch
import torch.nn as nn
import pandas as pd

from torch.utils.data import Dataset, DataLoader
from torchvision.models import mobilenet_v3_small

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


def build_model():
	model = mobilenet_v3_small(weights=None)
	in_feats = model.classifier[-1].in_features
	model.classifier[-1] = nn.Linear(in_feats, 1) # -- LogReg al final
	return model

def eval(logits, y_true, umbral=0.5):
	probs 	= torch.sigmoid(logits)
	y_pred 	= (probs >= umbral).float()
	y_true 	= y_true.float()

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

def main():
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	
	df = pd.read_csv(CSV_PATH)
	#df = df.dropna(subset=["diagnostic"]).reset_index(drop=True)
	df = df.dropna(subset=["diagnostic", IMAGE_COL]).reset_index(drop=True)

	print("1) filas tras dropna:", len(df))
	print("2) cáncer en df completo:", df["diagnostic"].isin(CANCER).sum())
	print("3) valores únicos:", [repr(v) for v in df["diagnostic"].unique()])

	train_df, val_df = train_test_split(df, test_size=0.2, random_state=0)

	train_ds = ImageOnlyDataset(train_df, IMAGE_DIR, IMAGE_COL, IMG_SIZE)
	val_ds = ImageOnlyDataset(val_df, IMAGE_DIR, IMAGE_COL, IMG_SIZE)
	
	print("4) positivos en train_ds.labels:", train_ds.labels.sum(), "de", len(train_ds.labels))

	print(CANCER)
	train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=0)
	val_loader = DataLoader(val_ds, batch_size=BATCH, shuffle=False, num_workers=0)

	y_tr = train_ds.labels
	n_pos = y_tr.sum()
	n_neg = len(y_tr) - n_pos
	pos_weight = torch.tensor([n_neg / max(n_pos, 1)], device=device)
	#pos_weight = torch.tensor([len(y_tr) - y_tr.sum()/ max(y_tr.sum(), 1)], device=device)
	print(f"pos_weight = {pos_weight.item():.2f}")


	model = build_model().to(device)
	opt = torch.optim.Adam(model.parameters(), lr=LR)
	loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

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


if __name__=="__main__":
	main()