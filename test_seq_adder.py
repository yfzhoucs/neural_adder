import model.seq_adder
import data.dataset_seq
from torch.utils.data import DataLoader
import argparse

parser = argparse.ArgumentParser(description='Args for testing SeqAdder.')

parser.add_argument('--ckpt_path', type=str, default='./1bit_w_carry_range.pth',
                    help='Path to adder ckpt.')
parser.add_argument('--max_len', type=int, default=1000,
                    help='Max length of binary integers.')
parser.add_argument('--num_samples', type=int, default=1000,
                    help='Number of samples to test.')
parser.add_argument('--batch_size', type=int, default=128,
                    help='Batch size.')

args = parser.parse_args()

def init():
    config = {}
    config['model'] = model.seq_adder.SeqAdder(args.ckpt_path)
    dataset = data.dataset_seq.SeqAdderDataset(args.num_samples, args.max_len)
    config['dataloader'] = DataLoader(dataset, args.batch_size, shuffle=False)

    return config


def test(model, dataloader):
    total_num = 0
    correct_num = 0
    for i, (x1, x2, y, c) in enumerate(dataloader):
        y_pred, c_pred = model(x1, x2)
        y_pred[y_pred > 0.9] = 1
        y_pred[y_pred < 0.1] = 0
        c_pred[c_pred > 0.9] = 1
        c_pred[c_pred < 0.1] = 0
        correct_num += y_pred.eq(y).sum().item() + c_pred.eq(c).sum().item()
        total_num += y_pred.nelement() + c_pred.nelement()
    
    return correct_num / total_num


def main():
    config = init()
    acc = test(**config)
    print('acc:', acc)


if __name__ == "__main__":
    main()