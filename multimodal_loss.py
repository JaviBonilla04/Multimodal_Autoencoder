import torch.nn as nn
import torch.nn.functional as F 

def multimodal_loss(recon, b):
	loss_num = F.mse_loss(recon["numeric"], b["numeric"])
	loss_cat = sum(F.cross_entropy(logits, b["multiclass"][:, i]) for i, logits in enumerate(recon["multiclass"])) / len(recon["multiclass"])
	loss_bin = F.binary_cross_entropy_with_logits(recon["binary"], b["binary"])
	loss_img = F.mse_loss(recon["image"], b["image"])

	print(f"num={loss_num.item():.3f}")
	print(f"cat={loss_cat.item():.3f}")
	print(f"bin={loss_bin.item():.3f}")
	print(f"img={loss_img.item():.3f}")

	return loss_num + loss_cat + loss_bin + loss_img