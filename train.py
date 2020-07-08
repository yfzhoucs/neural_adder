import data
import model
import data.dataset as dataset
import model.adder as adder
from torch.utils.data import DataLoader
import torch.nn as nn
import torch


def train_epoch(epoch_idx, model, data_loader, criterion, optimizer):
	model.train()

	for iter_idx, (x, y, c) in enumerate(data_loader):
		optimizer.zero_grad()
		y_pred, c_pred = model(x)
		loss = criterion(y_pred, y) + criterion(c_pred, c)
		# loss = criterion(c_pred, c)
		# loss = criterion(y_pred, y)
		loss.backward()
		optimizer.step()
		y_pred[y_pred > 0.5] = 1
		y_pred[y_pred <= 0.5] = 0
		c_pred[c_pred > 0.5] = 1
		c_pred[c_pred <= 0.5] = 0
		acc = (y_pred.eq(y).sum().item() + c_pred.eq(c).sum().item()) / (y_pred.nelement() + c_pred.nelement())
		# acc = (y_pred.eq(y).sum().item()) / (y_pred.nelement())
		if iter_idx % 10 == 0:
			print(epoch_idx, '{}/{}'.format(iter_idx, len(data_loader)), loss.item(), acc)

		if iter_idx == len(data_loader) - 1:
			print(x[0], y_pred[0], c_pred[0])


def train(epoch_num, model, data_loader, criterion, optimizer):
	for i in range(epoch_num):
		train_epoch(i, model, data_loader, criterion, optimizer)


def init(epoch_num, len_dataset, batch_size, bits):
	model = adder.Adder(bits=bits)
	adder_dataset = dataset.AdderDataset(len_dataset, bits=bits)
	dataloader = DataLoader(adder_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
	criterion = nn.MSELoss()
	optimizer = torch.optim.Adam(model.parameters())
	return epoch_num, model, dataloader, criterion, optimizer


def main():
	opt = init(500, 1000, 20, 2)
	train(*opt)
	torch.save(opt[1].state_dict(), './2bits.pth')


if __name__ == '__main__':
	main()