from torch.utils.data import Dataset
import numpy as np
from sklearn import preprocessing


class RamanDataset(Dataset):
    def __init__(self, path, train_data_fold, train=True):
        x = np.load(path + '/spectrum.npy').astype(np.float)
        y = np.load(path + '/components.npy').astype(np.float)
        assert (len(x) == len(y))
        fold_pos = int(len(x) * train_data_fold)
        # Pre-processing Xs:============================================================================================
        x = x[:fold_pos] if train else x[fold_pos:]
        # standard Gaussian scalar:
        scaler = preprocessing.StandardScaler().fit(x)
        self.x = scaler.transform(x)
        # Pre-processing Ys:============================================================================================
        self.y = y[:, :fold_pos] if train else y[:, fold_pos:]

    def __getitem__(self, item):
        return self.x[item], self.y[:, item]

    def __len__(self):
        return len(self.x)
