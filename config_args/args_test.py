import torch

class argparse():
    def __init__(self):
        self.epochs = 20
        self.learning_rate = 0.001
        self.patience = 4

        self.hidden_size = 40
        self.input_size = 30

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#args = argparse()
#args.epochs, args.learning_rate, args.patience = [30, 0.001, 4]
#args.hidden_size, args.input_size= [40, 30]
#args.device, = [torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),]
