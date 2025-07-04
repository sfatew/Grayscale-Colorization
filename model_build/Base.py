import torch.nn as nn
from torchvision import transforms

class BaseColor(nn.Module):
	def __init__(self):
		super(BaseColor, self).__init__()

		self.l_cent = 0.
		self.l_norm = 100.
		self.ab_norm = 110.

	def normalize_l(self, in_l):
		return (in_l-self.l_cent)/self.l_norm   # Normalize L to [-0.5, 0.5]

	def unnormalize_l(self, in_l):
		return in_l*self.l_norm + self.l_cent

	def normalize_ab(self, in_ab):
		return in_ab/self.ab_norm    # Normalize ab to [-1, 1]

	def unnormalize_ab(self, in_ab):
		return in_ab*self.ab_norm

	def imageNet_normalize(self, in_l):
		'''
		in_l: Tensor of shape (B, 1, H, W)
		return: Tensor of shape (B, 3, H, W)

		'''
		l_rgb = in_l.repeat(1,3,1,1)               # Gray to RGB

		# Define transform (example: just Normalize)
		normalize = transforms.Normalize(mean=[0.485,0.456,0.406],
										std=[0.229,0.224,0.225])

		# Apply directly
		img_normalized = normalize(l_rgb)
		return img_normalized
