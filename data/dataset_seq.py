from torch.utils.data import Dataset
import random
import torch

class SeqAdderDataset(Dataset):
    def __init__(self, num_samples, max_len):
        self.num_samples = num_samples
        self.max_len = max_len
        self.max_num = 2 ** max_len - 1

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        x1_int_bin = torch.randint(0, 2, (self.max_len,))
        x2_int_bin = torch.randint(0, 2, (self.max_len,))
        x1 = x1_int_bin.float() * 0.9 + torch.rand((1,)) * 0.1
        x2 = x2_int_bin.float() * 0.9 + torch.rand((1,)) * 0.1
        x1_int_str = ''.join(str(x) for x in x1_int_bin.numpy())[::-1]
        x2_int_str = ''.join(str(x) for x in x2_int_bin.numpy())[::-1]
        x1_int = int(x1_int_str, 2)
        x2_int = int(x2_int_str, 2)
        
        y_int = x1_int + x2_int
        y_int_str = bin(y_int)[2:].zfill(self.max_len)[::-1]
        cout_int = 1 if y_int > self.max_num else 0
        
        y = torch.zeros(self.max_len,)
        cout = torch.zeros(1)
        for i in range(self.max_len):
            y[i] = int(y_int_str[i])
        cout[0] = cout_int

        return x1, x2, y, cout


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    dataset = SeqAdderDataset(20, 10)
    dataloader = DataLoader(dataset, batch_size=10, shuffle=False, num_workers=0)
    for x in dataloader:
        print(x)
