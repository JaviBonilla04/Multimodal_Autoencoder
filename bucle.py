from torch.utils.data import DataLoader
import torch.nn as nn
import torch
from torch.utils.data import Dataset

import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

from multimodal_autoencoder import MultimodalAutoencoder
from dataset_handler import MultimodalDataset, MultimodalCSVDataset, Preprocessor, build_schema
from multimodal_loss import multimodal_loss

import pickle

IMG_DIR = "C:\\Users\\JB\\Desktop\\ML_Proyectos\\multimodal_learning\\autoencoder\\datasets\\imgs_part_1\\imgs_part_1"

device 	= torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(): 
	df = pd.read_csv("datasets\\metadata.csv")

	# -- all of the columns except the diagnostic
	numeric_cols = ["age", "fitspatrick", "diameter_1", "diameter_2"]
	multiclass_cols = ["background_father", "background_mother", "region", "itch", "grew", "hurt", "changed", "bleed", "elevation"]
	binary_cols = ["smoke", "drink", "pesticide", "gender", "skin_cancer_history", "cancer_history", "has_piped_water", "has_sewage_system"]
	image_col = "img_id"

	df = df.dropna(subset=binary_cols).reset_index(drop=True)

	for col in binary_cols:
		vals = df[col].dropna().unique()
		n_con_missing = len(vals) + (1 if df[col].isna().any() else 0)
		print(f"{col}: valores={sorted(vals.tolist())} | con missing serían {n_con_missing}")


	train_df, val_df = train_test_split(df, test_size=0.2, random_state=0)

	pre = Preprocessor(numeric_cols, multiclass_cols, binary_cols)
	pre.fit(train_df)

	train_ds 	= MultimodalCSVDataset(train_df, pre, IMG_DIR, image_col, img_size=32)
	val_ds 		= MultimodalCSVDataset(val_df, pre, IMG_DIR, image_col, img_size=32)

	# -- [TODO] set the workers to 4
	train_loader	= DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=0, pin_memory=True)
	val_loader		= DataLoader(val_ds, batch_size=64, shuffle=True, num_workers=0, pin_memory=True)

	schema = build_schema(pre, img_channels=3, img_size=32)
	print(schema)

	# -- fill the data with the amount of numeric columns
	model = MultimodalAutoencoder(
		n_numeric 			= schema.n_numeric,	
		multiclass_cards 	= schema.multiclass_cardinalities,
		n_binary 			= schema.n_binary,
		img_ch 				= schema.img_channels,
		img_size 			= schema.image_size,
		latent_dim 			= 32
	).to(device)

	opt = torch.optim.Adam(model.parameters(), lr=1e-3)

	model.train()
	for epoch in range(1, 11):
		total = 0.0
		for b in train_loader:
			b = {k: v.to(device) for k, v in b.items()}
			opt.zero_grad()
			recon, z = model(b)
			loss = multimodal_loss(recon, b)
			loss.backward()
			opt.step()
			total += loss.item()
		print(f"epoch {epoch:2d} | loss {total/len(train_loader):.4f}")

	model.eval()
	with torch.no_grad():
		b = {k: v[:4].to(device) for k, v in next(iter(val_loader)).items()}
		z = model.encode(b)
		print("forma del latente:", tuple(z.shape))

	torch.save(model.state_dict(), "trained_models/autoencoder1.pt")
	with open("Preprocessor.pkl", "wb") as f:
		pickle.dump(pre, f)

if __name__=="__main__":
	main()

