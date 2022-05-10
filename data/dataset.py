from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, data, his_len=96, pre_len=96):
        self.data = data
        self.his_len = his_len
        self.pre_len = pre_len

    def __getitem__(self, index):
        X = self.data[index:index + self.his_len, :]
        y = self.data[index + self.his_len: index + self.his_len + self.pre_len, :]
        return X, y

    def __len__(self):
        return len(self.data) - self.his_len - self.pre_len + 1
