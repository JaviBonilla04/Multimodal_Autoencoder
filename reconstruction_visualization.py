import torch
import numpy as np
import matplotlib.pyplot as plt
import pickle


def _to_hwc(img_tensor):
	arr = img_tensor.detach().cpu().numpy()
	if arr.shape[0] == 1:
		return arr[0]
	return np.transpose(arr, (1, 2, 0))


def visaulize_recon(model, dataset, device, n=8, save=None, seed=0):
	rng = np.random.default_rng(seed)
	idxs = rng.choice(len(dataset), size=min(n, len(dataset)), replace=False)

	samples = [dataset[i] for i in idxs]
	batch = {
		k: torch.stack([m[k] for m in samples]).to(device)
		for k in samples[0].keys()
	}

	model.eval()
	with torch.no_grad():
		recon, z = model(batch)

	orig_imgs 	= batch["image"].cpu()
	recon_imgs 	= recon["image"].cpu().clamp(0, 1)


	errors = ((orig_imgs - recon_imgs)**2).mean(dim=[1, 2, 3]).numpy()

	fig, axes = plt.subplots(2, n, figsize=(1.6*n, 3.6), squeeze=False)

	for j in range(n):
		axes[0, j].imshow(
			_to_hwc(orig_imgs[j]), 
			#cmap="gray" if orig_imgs[j].shape[0] == 1 else None
			interpolation="nearest"
		)
		axes[0, j].axis("off")

		axes[1, j].imshow(
			_to_hwc(recon_imgs[j]),
			#cmap="gray" if recon_imgs[j].shape[0] == 1 else None
			interpolation="nearest"
		)
		axes[1, j].axis("off")
		axes[1, j].set_title(f"mse={errors[j]:.3f}", fontsize=8)

	axes[0, 0].set_ylabel("ORIGINAL", fontsize=10)
	axes[1, 0].set_ylabel("RECON", fontsize=10)

	fig.text(0.02, 0.72, "ORIGINAL", rotation=90, va="center", fontsize=11, weight="bold")
	fig.text(0.02, 0.28, "RECON", rotation=90, va="center", fontsize=11, weight="bold")

	plt.suptitle("Orginal vs Recon Autoencoder", fontsize=12)
	plt.tight_layout()

	if save:
		plt.savefig(save, dpi=150, bbox_inches="tight")
		print(f"Saved in {save}")
	plt.show()
	return errors


if __name__=="__main__":

	IMG_DIR		= "C:\\Users\\JB\\Desktop\\ML_Proyectos\\multimodal_learning\\autoencoder\\datasets\\imgs_part_1\\imgs_part_1"
	WEIGHTS 	= "trained_models\\autoencoder1.pt"

	import pandas as pd
	from multimodal_autoencoder import MultimodalAutoencoder
	from dataset_handler import Preprocessor, MultimodalCSVDataset, build_schema

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	df = pd.read_csv("datasets\\metadata.csv")

	numeric_cols = ["age", "fitspatrick", "diameter_1", "diameter_2"]
	multiclass_cols = ["background_father", "background_mother", "region", "itch", "grew", "hurt", "changed", "bleed", "elevation"]
	binary_cols = ["smoke", "drink", "pesticide", "gender", "skin_cancer_history", "cancer_history", "has_piped_water", "has_sewage_system"]

	with open("preprocessor.pkl", "rb") as f:
		pre = pickle.load(f)

	#pre = Preprocessor(numeric_cols, multiclass_cols, binary_cols).fit(df)
	sch = build_schema(pre, 3, 32)

	ds = MultimodalCSVDataset(df, pre, IMG_DIR, "img_id", img_size=32)

	model = MultimodalAutoencoder(
		sch.n_numeric, sch.multiclass_cardinalities, sch.n_binary, 3, 32, 32
	).to(device)

	model.load_state_dict(
		torch.load(WEIGHTS, map_location=device)
	)

	errors = visaulize_recon(model, ds, device, n=6, save="recon_demo.png")
	print("MSE by image:", [f"{e:.4f}" for e in errors])