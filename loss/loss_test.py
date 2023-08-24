import torch

class TestLoss(torch.nn.Module):
    
    def __init__(self):
        super(TestLoss, self).__init__()

    def forward(self, output, target):

        hinge_loss = 1 - torch.mul(output, target)
        hinge_loss[hinge_loss < 0] = 0
        
        return torch.mean(hinge_loss)