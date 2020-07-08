import torch.nn as nn
import torch.nn.functional as F
import torch

class Adder(nn.Module):
	def __init__(self):
		super(Adder, self).__init__()
		self.linear1 = nn.Linear(3, 6)
		# self.linear2 = nn.Linear(bits * 8, bits * 8)
		# self.linear3 = nn.Linear(bits * 8, bits * 8)
		# self.linear3 = nn.Linear(320, 320)
		# self.linear4 = nn.Linear(320, 64)
		# self.linear5 = nn.Linear(64, 8)
		self.out_y = nn.Linear(6, 1)
		self.out_c = nn.Linear(6, 1)

	def forward(self, x1, x2, cin):
		x1 = x1.float()
		x2 = x2.float()
		cin = cin.float()
		# print(x1.shape, x2.shape, cin.shape)
		x = torch.cat([x1, x2, cin], dim=1)
		x = F.sigmoid(self.linear1(x))
		# x = F.sigmoid(self.linear2(x))
		# x = F.sigmoid(self.linear3(x))
		y = F.sigmoid(self.out_y(x))
		cout = F.sigmoid(self.out_c(x))
		return y, cout


if __name__ == '__main__':
	import torch
	adder = Adder()
	x1 = torch.randint(0, 2, (1, 1))
	x2 = torch.randint(0, 2, (1, 1))
	cin = torch.randint(0, 2, (1, 1))
	y, cout = adder(x1, x2, cin)
	print(x1, x2, cin)
	print(y)
	print(cout)