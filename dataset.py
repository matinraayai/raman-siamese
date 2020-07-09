from torch.utils.data import Dataset
import numpy as np
from sklearn import preprocessing


class RamanDataset(Dataset):
    def __init__(self, path, train_data_fold, train=True):
        x = np.load(path + '/spectrum.npy').astype(np.float32)
        y = np.load(path + '/components.npy')
        assert (x.shape[0] == y.shape[1])
        fold_pos = int(len(x) * train_data_fold)
        # Pre-processing Xs:============================================================================================
        x = x[:fold_pos] if train else x[fold_pos:]
        # standard Gaussian scalar:
        scaler = preprocessing.StandardScaler().fit(x)
        self.x = scaler.transform(x).reshape(-1, 1, 881)
        # Pre-processing Ys:============================================================================================
        self.y = y[:, :fold_pos] if train else y[:, fold_pos:]

    def __getitem__(self, item):
        idx = np.random.choice(self.x.shape[0], 2, replace=False)
        x_1, x_2 = self.x[idx[0]], self.x[idx[1]]
        y_1, y_2 = self.y[:, idx[0]], self.y[:, idx[1]]
        label = (y_1 * y_2).sum().astype(np.bool).astype(np.float32)
        return x_1, x_2, np.array([label])

    def __len__(self):
        return len(self.x)
