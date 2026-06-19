import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split


from multimodal_autoencoder import MultimodalAutoencoder
from dataset_handler import Preprocessor, MultimodalCSVDataset, build_schema


import pickle

CSV_PATH 	= "C:\\Users\\JB\\Desktop\\ML_Proyectos\\multimodal_learning\\autoencoder\\datasets\\metadata.csv"
IMAGE_DIR 	= "C:\\Users\\JB\\Desktop\\ML_Proyectos\\multimodal_learning\\autoencoder\\datasets\\imgs_part_1\\imgs_part_1"


WEIGHTS 	= "trained_models\\autoencoder1.pt"
IMG_SIZE 	= 32
LATENT_DIM 	= 32 


NUMERIC_COLS 	= ["age", "fitspatrick", "diameter_1", "diameter_2"]
MULTICLASS_COLS = ["background_father", "background_mother", "region", "itch", "grew", "hurt", "changed", "bleed", "elevation"]
BINARY_COLS 	= ["smoke", "drink", "pesticide", "gender", "skin_cancer_history", "cancer_history", "has_piped_water", "has_sewage_system"]
IMAGE_COL 		= "img_id"

CANCER = {"BBC", "MEL", "SCC"} # -- the categories on the diagnostic column that represent a type of cancer

def extract_lantent(df, pre, model, device):
	ds 		= MultimodalCSVDataset(df, pre, IMAGE_DIR, IMAGE_COL, img_size=IMG_SIZE)
	loader 	= DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)

	Z = []
	model.eval()
	with torch.no_grad():
		for b in loader:
			b = {k: v.to(device) for k, v in b.items()}
			z = model.encode(b)
			Z.append(z.cpu())
	Z = torch.cat(Z, dim=0)
	y = df["diagnostic"].isin(CANCER).astype(np.float32).values
	y = torch.from_numpy(y)
	return Z, y

class LogisticRegression(nn.Module):
	def __init__(self, in_dim):
		super().__init__()
		self.linear = nn.Linear(in_dim, 1)
	
	def forward(self, z):
		return self.linear(z).squeeze(1)

def evaluate(logits, y_true, umbral=0.5):
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
		"acc"	: acc,
		"prec"	: prec,
		"rec"	: rec,
		"f1"	: f1,
		"tp"	: tp,
		"tn"	: tn,
		"fp"	: fp,
		"fn"	: fn,
	}


def main():
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	
	
	df = pd.read_csv(CSV_PATH)
	df = df.dropna(subset=NUMERIC_COLS + MULTICLASS_COLS).reset_index(drop=True)
	
	train_df, val_df = train_test_split(df, test_size=0.2, random_state=0)
	train_df 	= train_df.reset_index(drop = True)
	val_df		= val_df.reset_index(drop=True)

	#pre = Preprocessor(NUMERIC_COLS, MULTICLASS_COLS, BINARY_COLS).fit(train_df)
	

	with open("preprocessor.pkl", "rb") as f:
		pre = pickle.load(f)

	print("cardinalidades clasificador:", pre.multiclass_cardinalities)

	schema = build_schema(pre, img_channels=3, img_size=IMG_SIZE)

	model = MultimodalAutoencoder(
		n_numeric			= schema.n_numeric,
		multiclass_cards 	= schema.multiclass_cardinalities,
		n_binary			= schema.n_binary,
		img_ch 				= schema.img_channels,
		img_size			= schema.image_size,
		latent_dim			= LATENT_DIM,
	).to(device)

	model.load_state_dict(
		torch.load(WEIGHTS, map_location=device)
	)

	Z_tr, y_tr = extract_lantent(train_df, pre, model, device)
	Z_va, y_va = extract_lantent(val_df, pre, model, device)

	Z_tr, y_tr = Z_tr.to(device), y_tr.to(device)
	Z_va, y_va = Z_va.to(device), y_va.to(device)

	print(f"z train: {tuple(Z_tr.shape)} | z val: {tuple(Z_va.shape)}")
	print(f"cáncer en train: {y_tr.mean().item():.1%} | en val: {y_va.mean().item():.1%}")

	n_pos = y_tr.sum().item()
	n_neg = len(y_tr) - n_pos
	pos_weight = torch.tensor([n_neg / max(n_pos, 1)], device=device)
	print(f"pos_weight (desbalance): {pos_weight.item():.2f}")

	clf = LogisticRegression(LATENT_DIM).to(device)
	opt = torch.optim.Adam(clf.parameters(), lr=1e-2)
	loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

	clf.train()

	for epoch in range(1, 101):
		opt.zero_grad()
		logits = clf(Z_tr)
		loss = loss_fn(logits, y_tr)
		loss.backward()
		opt.step()
		if epoch % 20 == 0: print(f"epoch: {epoch:3d} | loss {loss.item():.4f}")

	clf.eval()
	with torch.no_grad():
		logits_va = clf(Z_va)
	m = evaluate(logits_va.cpu(), y_va.cpu())
	
	print("\n" + "=" * 50)
	print("VAL RESULST")
	print(f"	Accuracy	:	{m['acc']:.3f}")
	print(f"	Precision	: 	{m['prec']:.3f}")
	print(f"	Recall		: 	{m['rec']:.3f}")
	print(f"	F1			: 	{m['f1']:.3f}")
	print(f"\n Confusion Matrix:			")
	print(f"                 	 pred neg   		pred pos")
	print(f"    real neg  |    {m['tn']:5d}      {m['fp']:5d}")
	print(f"    real pos  |    {m['fn']:5d}      {m['tp']:5d}")


if __name__=="__main__":
	main()