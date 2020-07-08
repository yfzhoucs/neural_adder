import torch.nn as nn
import torch.nn.functional as F

class Adder(nn.Module):
	def __init__(self, bits=4):
		super(Adder, self).__init__()
		self.bits = bits
		self.linear1 = nn.Linear(bits * 2, bits * 8)
		# self.linear2 = nn.Linear(bits * 8, bits * 8)
		# self.linear3 = nn.Linear(bits * 8, bits * 8)
		# self.linear3 = nn.Linear(320, 320)
		# self.linear4 = nn.Linear(320, 64)
		# self.linear5 = nn.Linear(64, 8)
		self.out_y = nn.Linear(bits * 8, bits)
		self.out_c = nn.Linear(bits * 8, 1)

	def forward(self, x):
		x = x.float()
		x = x.view(-1, self.bits * 2)
		x = F.sigmoid(self.linear1(x))
		# x = F.sigmoid(self.linear2(x))
		# x = F.sigmoid(self.linear3(x))
		y = F.sigmoid(self.out_y(x))
		c = F.sigmoid(self.out_c(x))
		return y, c


if __name__ == '__main__':
	import torch
	adder = Adder()
	x = torch.randint(0, 2, (1, 2, 4))
	y, c = adder(x)
	print(y)
	print(c)