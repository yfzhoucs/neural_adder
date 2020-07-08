from torch.utils.data import Dataset
import random
import torch

class AdderDataset(Dataset):
	def __init__(self, num_samples, bits=4):
		self.num_samples = num_samples
		self.bits = bits

	def __len__(self):
		return self.num_samples

	def __getitem__(self, idx):
		num_1 = random.randint(0, 2 ** self.bits - 1)
		num_2 = random.randint(0, 2 ** self.bits - 1)
		num_c = 1 if num_1 + num_2 >= 2 ** self.bits else 0
		num_y = '{0:b}'.format((num_1 + num_2) % (2 ** self.bits)).zfill(self.bits)
		num_1 = '{0:b}'.format(num_1).zfill(self.bits)
		num_2 = '{0:b}'.format(num_2).zfill(self.bits)
		x = torch.zeros(2, self.bits)
		y = torch.zeros(self.bits)
		c = torch.zeros(1)

		for i in range(self.bits):
			x[0][i] = int(num_1[i])
			x[1][i] = int(num_2[i])
			y[i] = int(num_y[i])
		c[0] = num_c

		return x, y, c




if __name__ == '__main__':
	from torch.utils.data import DataLoader
	dataset = AdderDataset(20, 1)
	dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
	for x in dataloader:
		print(x)
