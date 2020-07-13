from torch.utils.data import Dataset
import numpy as np
from sklearn import preprocessing


class RamanDataset(Dataset):
    def __init__(self, path, train_data_fold, train=True):
        super(RamanDataset, self).__init__()
        x = np.load(path + '/spectrum.npy').astype(np.float32)
        y = np.load(path + '/components_naveen.npy')
        assert (x.shape[0] == y.shape[0])
        fold_pos = int(len(x) * train_data_fold)
        # Pre-processing Xs:============================================================================================
        x = x[:fold_pos] if train else x[fold_pos:]
        # standard Gaussian scalar:
        scaler = preprocessing.StandardScaler().fit(x)
        self.x = scaler.transform(x).reshape(-1, 1, 881)
        # Pre-processing Ys:============================================================================================
        self.y = y[:fold_pos] if train else y[fold_pos:]
        # Finding pure components in the dataset:=======================================================================
        pure_idx = set()
        for i, entry in enumerate(self.y):
            if entry.sum() == 1:
                pure_idx.add(i)
        self.pure_idx = list(pure_idx)

    def __getitem__(self, item):
        # Randomly select a pure component and its signal
        pure_component_idx = np.random.choice(len(self.pure_idx))
        mixture = self.x[item]
        mixture_label = self.y[item]
        pure_component = self.x[pure_component_idx]
        pure_component_label = self.y[pure_component_idx].argmax()
        # If the component is present in the mixture, the label is 1.0; Else, 0.0
        label = np.array([mixture_label[pure_component_label] == 1], dtype=np.float32)
        return pure_component, mixture, label

    def __len__(self):
        return len(self.x)


class RamanDataset2(RamanDataset):
    def __init__(self, path, train_data_fold, train=True):
        super(RamanDataset2, self).__init__(path, train_data_fold, train)
        self.x = self.x.reshape(-1, 881, 1)

    def __getitem__(self, item):
        # Randomly select a pure component and its signal
        pure_component, mixture, label = super().__getitem__(item)
        return np.stack((pure_component, mixture)).reshape(1, 881, 2), label

    def __len__(self):
        return len(self.x)
