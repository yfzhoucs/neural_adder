import adder
import torch.nn as nn
import torch.nn.functional as F
import torch


class SeqAdder(nn.Module):
    def __init__(self, ckpt_path):
        super(SeqAdder, self).__init__()
        self.ckpt_path = ckpt_path
        self.adder = adder.Adder()
        if self.ckpt_path:
            self.adder.load_state_dict(
                torch.load(self.ckpt_path))
    
    def forward(self, x1, x2):
        assert x1.shape[1] == x2.shape[1]
        length = x1.shape[1]
        y = torch.zeros(*x1.shape)

        c = torch.zeros(x1.shape[0], 1)
        for i in range(length):
            print(x1[:, i:i+1].shape)
            y_i, c = self.adder(x1[:,i:i+1], x2[:,i:i+1], c)
            y[:,i:i+1] = y_i
        
        return y, c


if __name__ == "__main__":
    seq_adder = SeqAdder('./1bit_w_carry.pth')
    x1 = torch.randint(0, 2, (5, 10))
    x2 = torch.randint(0, 2, (5, 10))
    y, c = seq_adder(x1, x2)
    
    y[y > 0.5] = 1
    y[y <= 0.5] = 0
    c[c > 0.5] = 1
    c[c <= 0.5] = 0
    for i in range (x1.shape[0]):
        print(x1[i], x2[i], c[i], y[i])