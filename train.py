import data
import model
import data.dataset as dataset
import model.adder as adder
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import argparse


parser = argparse.ArgumentParser(description='Args for training.')

parser.add_argument('--epoch_num', type=int, default=1000,
                    help='Number of epochs to train.')
parser.add_argument('--len_dataset', type=int, default=1000,
                    help='Number of iters in an epoch.')
parser.add_argument('--batch_size', type=int, default=20,
                    help='Batch size.')
parser.add_argument('--output_path', type=str, default='./1bit_w_carry_range.pth',
                    help='Path to put the model ckpt.')

args = parser.parse_args()


def train_epoch(epoch_idx, model, data_loader, criterion, optimizer):
	model.train()

	for iter_idx, (x1, x2, cin, y, cout) in enumerate(data_loader):
		optimizer.zero_grad()
		y_pred, cout_pred = model(x1, x2, cin)
		loss = criterion(y_pred, y) + criterion(cout_pred, cout)

		loss.backward()
		optimizer.step()
		y_pred[y_pred > 0.9] = 1
		y_pred[y_pred <= 0.1] = 0
		cout_pred[cout_pred > 0.9] = 1
		cout_pred[cout_pred <= 0.1] = 0
		acc = (y_pred.eq(y).sum().item() + cout_pred.eq(cout).sum().item()) / (y_pred.nelement() + cout_pred.nelement())

		if iter_idx % 10 == 0:
			print(epoch_idx, '{}/{}'.format(iter_idx, len(data_loader)), loss.item(), acc)

		if iter_idx == len(data_loader) - 1:
			print(x1[0], x2[0], cin[0], y_pred[0], cout_pred[0])


def train(epoch_num, model, data_loader, criterion, optimizer):
	for i in range(epoch_num):
		train_epoch(i, model, data_loader, criterion, optimizer)


def init(epoch_num, len_dataset, batch_size):
	model = adder.Adder()
	adder_dataset = dataset.AdderDataset(len_dataset)
	dataloader = DataLoader(adder_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
	criterion = nn.MSELoss()
	optimizer = torch.optim.Adam(model.parameters())
	return epoch_num, model, dataloader, criterion, optimizer


def main():
	opt = init(args.epoch_num, args.len_dataset, args.batch_size)
	train(*opt)
	torch.save(opt[1].state_dict(), args.output_path)


if __name__ == '__main__':
	main()