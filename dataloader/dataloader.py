import torch
from torch.utils.data import Dataset, DataLoader

class data_test(Dataset):
    def __init__(self, flag='train'):
        assert flag in ['train', 'test', 'valid']
        self.flag = flag
        self.__load_data__()

    def __getitem__(self, index):
        pass
    def __len__(self):
        pass

    def __load_data__(self, csv_paths: list='test'):
        pass
        print(
            "train_X.shape:{}\ntrain_Y.shape:{}\nvalid_X.shape:{}\nvalid_Y.shape:{}\n"
            .format(self.train_X.shape, self.train_Y.shape, self.valid_X.shape, self.valid_Y.shape))

#train_dataset = data_test(flag='train')
#train_dataloader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
#valid_dataset = data_test(flag='valid')
#valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=64, shuffle=True)