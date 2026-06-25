import torch.nn as nn
import torch.nn.functional as F 

import matplotlib.pyplot as plt

from typing import List, Dict
from dataclasses import dataclass

@dataclass
class LossSchema:
	loss_num 	: float
	loss_cat 	: float
	loss_bin 	: float
	loss_img 	: float	

def build_loss_shema(loss_num, loss_cat, loss_bin, loss_img):
	return LossSchema(
		loss_num 	= loss_num.item(),
		loss_cat 	= loss_cat.item(),
		loss_bin 	= loss_bin.item(),
		loss_img 	= loss_img.item(),
	)

def multimodal_loss(recon, b):
	loss_num = F.mse_loss(recon["numeric"], b["numeric"])
	loss_cat = sum(F.cross_entropy(logits, b["multiclass"][:, i]) for i, logits in enumerate(recon["multiclass"])) / len(recon["multiclass"])
	loss_bin = F.binary_cross_entropy_with_logits(recon["binary"], b["binary"])
	loss_img = F.mse_loss(recon["image"], b["image"])

	print(f"num={loss_num.item():.3f}")
	print(f"cat={loss_cat.item():.3f}")
	print(f"bin={loss_bin.item():.3f}")
	print(f"img={loss_img.item():.3f}")

	loss_schema = build_loss_shema(loss_num, loss_cat, loss_bin, loss_img)


	return loss_num + loss_cat + loss_bin + 10*loss_img, loss_schema


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


