from torch.utils.data import Dataset
import random
import torch

class AdderDataset(Dataset):
	def __init__(self, num_samples):
		self.num_samples = num_samples

	def __len__(self):
		return self.num_samples

	def __getitem__(self, idx):
		x1_int = torch.randint(0, 2, (1,))
		x2_int = torch.randint(0, 2, (1,))
		cin_int = torch.randint(0, 2, (1,))
		x1 = x1_int.float() * 0.9 + torch.rand((1,)) * 0.1
		x2 = x2_int.float() * 0.9 + torch.rand((1,)) * 0.1
		cin = cin_int.float() * 0.9 + torch.rand((1,)) * 0.1
		
		sum_of_inputs = x1_int.item() + x2_int.item() + cin_int.item()
		
		y = torch.zeros(1)
		cout = torch.zeros(1)
		y[0] = sum_of_inputs % 2
		cout[0] = 1 if sum_of_inputs >= 2 else 0

		return x1, x2, cin, y, cout


if __name__ == '__main__':
	from torch.utils.data import DataLoader
	dataset = AdderDataset(20)
	dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
	for x in dataloader:
		print(x)
