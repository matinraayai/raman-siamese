from torch.utils.data import Dataset
import numpy as np
from sklearn import preprocessing


class RamanTrain(Dataset):
    def __init__(self, path, compound, fold):
        x = np.load(path + "/" + str(compound) + 'component.npy').astype(np.float)
        y = np.load(path + "/" + str(compound) + 'label.npy').astype(np.float)
        assert (len(x) == len(y))
        train_fold_pos = int(len(x) * fold)
        # Pre-processing Xs:============================================================================================
        x = x[:train_fold_pos]
        # standard Gaussian scalar:
        scaler = preprocessing.StandardScaler().fit(x)
        self.x = scaler.transform(x)
        # Pre-processing Ys:============================================================================================
        y_neg = np.ones(y.shape) - y
        y = np.concatenate((y, y_neg), axis=1)
        self.y = y[:train_fold_pos]

    def __getitem__(self, item):
        return self.x[item], self.y[item]

    def __len__(self):
        return len(self.x)


class RamanTest(Dataset):
    def __init__(self, path, compound, fold):
        x = np.load(path + "/" + str(compound) + 'component.npy').astype(np.float)
        y = np.load(path + "/" + str(compound) + 'label.npy').astype(np.float)
        assert (len(x) == len(y))
        test_fold_pos = int(len(x) * fold)
        # Pre-processing Xs:========================================================================================
        x = x[test_fold_pos:]
        # standard Gaussian scalar:
        scaler = preprocessing.StandardScaler().fit(x)
        self.x = scaler.transform(x)
        # Pre-processing Ys:========================================================================================
        y_neg = np.ones(y.shape) - y
        y = np.concatenate((y, y_neg), axis=1)
        self.y = y[test_fold_pos:]

    def __getitem__(self, item):
        return self.x[item], self.y[item]

    def __len__(self):
        return len(self.x)
