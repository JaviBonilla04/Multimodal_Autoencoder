import torch.nn as nn
import torch
import torch.nn.functional as F 
from encoders import NumericEncoder, MulticlassEncoder, BinaryEncoder, ImageEncoder
from fusion_layer import Fusion
from decoders import NumericDecoder, MulticlassDecoder, BinaryDecoder, ImageDecoder



class MultimodalAutoencoder(nn.Module):
	def __init__(self, n_numeric, multiclass_cards, n_binary, img_ch=3, img_size=32, latent_dim=32):
		super().__init__()
		self.enc_num	= NumericEncoder(n_numeric)
		self.enc_cat	= MulticlassEncoder(multiclass_cards)
		self.enc_bin	= BinaryEncoder(n_binary)
		self.enc_img	= ImageEncoder(img_ch, img_size)

		fusion_in = (
			self.enc_num.out_dim +
			self.enc_cat.out_dim + 
			self.enc_bin.out_dim +
			self.enc_img.out_dim
		)

		self.fusion = Fusion(fusion_in, latent_dim)

		self.dec_num	= NumericDecoder(latent_dim, n_numeric)
		self.dec_cat	= MulticlassDecoder(latent_dim, multiclass_cards)
		self.dec_bin	= BinaryDecoder(latent_dim, n_binary)
		self.dec_img	= ImageDecoder(latent_dim, img_ch, self.enc_img.feat_hw)

	def encode(self, b):
		h = torch.cat([self.enc_num(b["numeric"]), self.enc_cat(b["multiclass"]), self.enc_bin(b["binary"]),  self.enc_img(b["image"])], dim=1)
		return self.fusion(h)

	def forward(self, b):
		z = self.encode(b)
		recon = {"numeric": self.dec_num(z), "multiclass": self.dec_cat(z), "binary": self.dec_bin(z),  "image": self.dec_img(z)}
		return recon, z

