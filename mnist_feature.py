'''
Created on Oct 23, 2017

@author: longxiang
'''

from torch.utils.data import Dataset
from mnist_sampler import *        

class MNISTFeature(Dataset):
    training_file = 'feature_train.pkl'
    test_file = 'feature_test.pkl'

    def __init__(self, root=config.data_dir, train=True, transform=None, target_transform=None):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set

        if not self._check_exists():
            raise RuntimeError('Dataset not found.')

        if self.train:
            self.train_data, self.train_labels, self.train_idxs = load_pkl(
                os.path.join(self.root, self.training_file))
        else:
            self.test_data, self.test_labels, self.test_idxs = load_pkl(
                os.path.join(self.root, self.test_file))

    def __getitem__(self, index):
        if self.train:
            feat, target, idx = self.train_data[index], self.train_labels[index], self.train_idxs[index]
        else:
            feat, target, idx = self.test_data[index], self.test_labels[index], self.test_idxs[index]

        if self.transform is not None:
            feat = self.transform(feat)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return feat, target, idx

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def _check_exists(self):
#         return True
        return os.path.exists(os.path.join(self.root, self.training_file)) and \
            os.path.exists(os.path.join(self.root, self.test_file))
    
            
if __name__ == '__main__':
    dataset = MNISTFeature(train=True)
    print len(dataset)
     
    dataset = MNISTFeature(train=False)
    print len(dataset)
