from torch.utils.data import Dataset
import numpy as np
from sklearn import preprocessing


class DeepCidDataset(Dataset):
    def __init__(self, path, train_data_fold, train=True):
        super(DeepCidDataset, self).__init__()
        x = np.load(path + '/spectrum.npy').astype(np.float32)
        y = np.load(path + '/components_naveen.npy')
        assert (x.shape[0] == y.shape[0])
        # Pre-processing Xs:============================================================================================
        # standard Gaussian scalar used in the DeepCID paper
        scaler = preprocessing.StandardScaler().fit(x)
        x = scaler.transform(x)
        x = x.reshape(-1, 1, x.shape[-1])
        # Finding pure components in the dataset:=======================================================================
        # They should be saved separately from the actual Xs so that both the train and test datasets have access to
        # them
        # The pure_component_table is in the form of (pure_comp_id, list(pure_component))
        self.pure_component_table = {}
        for i, entry in enumerate(y):
            if entry.sum() == 1:
                component_id = np.where(entry == 1)[0][0]
                if component_id not in self.pure_component_table.keys():
                    self.pure_component_table[component_id] = [x[i]]
                else:
                    self.pure_component_table[component_id].append(x[i])

        # Fold positions:===============================================================================================
        fold_pos = int(len(x) * train_data_fold)
        self.x = x[:fold_pos] if train else x[fold_pos:]
        self.y = y[:fold_pos] if train else y[fold_pos:]

    def __getitem__(self, item):
        # Uniformly select whether the final label will be 0 or 1 for balancing what the model sees through
        # training
        label = np.random.choice(2)
        mixture = self.x[item]
        mixture_label = self.y[item]

        # Randomly select a pure component based on the label
        possible_components = np.where(mixture_label == label)[0]
        # Change label if it's not present in the mixture label array
        if len(possible_components) == 0:
            label = int(not label)
            possible_components = np.where(mixture_label == label)[0]
        pure_component_idx = np.random.choice(possible_components)
        pure_component_array = self.pure_component_table[pure_component_idx]
        # Randomly select a pure component signal out of the component entry:
        pure_component = pure_component_array[np.random.choice(len(pure_component_array))]

        return pure_component, mixture, np.array([label], dtype=np.float32)

    def __len__(self):
        return len(self.x)


class DeepCidDataset2(DeepCidDataset):
    def __init__(self, path, train_data_fold, train=True):
        super(DeepCidDataset2, self).__init__(path, train_data_fold, train)
        self.x = self.x.reshape(-1, 881, 1)

    def __getitem__(self, item):
        # Randomly select a pure component and its signal
        pure_component, mixture, label = super().__getitem__(item)
        label_new = np.zeros(2).astype(np.float32)
        label_new[int(label)] = 1.0
        return np.stack((pure_component, mixture)).reshape(1, 881, 2), label_new

    def __len__(self):
        return len(self.x)
