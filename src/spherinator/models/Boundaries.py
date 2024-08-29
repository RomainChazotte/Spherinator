import torch

import numpy as np


class detect(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.Bound1 = torch.nn.Conv2d(3,1,3)
        self.Bound1.weight.data =torch.cat(( torch.tensor([[[[1,1,1],[0,0,0],[-1,-1,-1]]]],dtype=torch.float),torch.tensor([[[[1,1,1],[0,0,0],[-1,-1,-1]]]],dtype=torch.float),torch.tensor([[[[1,1,1],[0,0,0],[-1,-1,-1]]]],dtype=torch.float)),dim=1)
        self.Bound2 = torch.nn.Conv2d(3,1,3)
        print(self.Bound2.weight.data.size())
        self.Bound2.weight.data = torch.cat((torch.tensor([[[[1,0,-1],[1,0,-1],[1,0,-1]]]],dtype=torch.float),torch.tensor([[[[1,0,-1],[1,0,-1],[1,0,-1]]]],dtype=torch.float),torch.tensor([[[[1,0,-1],[1,0,-1],[1,0,-1]]]],dtype=torch.float)), dim=1)
        print(self.Bound2.weight.data.size())
        self.relu = torch.nn.ReLU()
        '''
        self.Bound3 = torch.nn.Conv2d(1,1,3)
        self.Bound3.weight = torch.tensor([[-1,-1,-1],[0,0,0],[1,1,1]])
        self.Bound4 = torch.nn.Conv2d(1,1,3)
        self.Bound4.weight = torch.tensor([[-1,0,1],[-1,0,1],[-1,0,1]])
        '''
    def forward(self,x):
        x = torch.tensor(x,dtype=torch.float)
        x1= self.relu((torch.abs(self.Bound1(x)))-0.5)

        x2= self.relu((torch.abs(self.Bound1(x)))-0.5)
        x = x1+x2

        return np.array(x.detach())

detector = detect()
input = np.zeros((3,27,27))
input[:,10:16,12:13]=1
print(detector(input))