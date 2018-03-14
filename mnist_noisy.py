
'''
Created on Oct 23, 2017

@author: longxiang
'''

import shutil
from torch.utils.data import Dataset
from mnist_sampler import *

np.random.seed(1234)

K_train = 5
K_test = 1
N_noisy_train = 6000 * K_train
N_noisy_test = 1000 * K_test
H = 28
W = 28

def generate_dataset(train=True, n_noisy=N_noisy_train, k=K_train):
    datas = []
    img_sampler = get_noisy_sampler()
    num_sampler = get_number_sampler(load_mnist(train))
    for i in range(10):
        print 'Generate for number %d ...'%i
        idx = 0
        for _ in range(k):
            for num in num_sampler.numbers[i]:
                img =  img_sampler.sample((H,W))   
                put_numbers(img, num)
                datas.append((img, i))
                idx += 1
                if idx%10000 == 0:
                    print idx
    print 'Generate for noisy ...'
    for idx in range(n_noisy):
        img =  img_sampler.sample((H,W))   
        datas.append((img, 10)) 
        if (idx+1)%10000 == 0:
            print idx+1
    return datas

def save_pkl(datas, path):
    print 'save pkl to: %s'%path
    train_imgs, train_labels = zip(*datas)
    dump_pkl((train_imgs, train_labels), path)
    
def save_images(datas, image_dir):
    print 'save images to: %s' %image_dir
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
    idx = 0
    for img, label in datas:
        img = to_image(img)
        img.save(os.path.join(image_dir, '%d_%06d.bmp'%(label, idx)))
        idx += 1


class MNISTNoisy(Dataset):
    training_images_root = 'noisy_train'
    test_images_root = 'noisy_test'
    training_file = 'noisy_train.pkl'
    test_file = 'noisy_test.pkl'

    def __init__(self, root=config.data_dir, train=True, transform=None, target_transform=None, generate=False, force_generate=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set

        if generate:
            self.generate(force_generate)

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use generate=True to generate it')

        if self.train:
            self.train_data, self.train_labels = load_pkl(
                os.path.join(self.root, self.training_file))
        else:
            self.test_data, self.test_labels = load_pkl(
                os.path.join(self.root, self.test_file))

    def __getitem__(self, index):
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        img = Image.fromarray(img, mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, '%d_%06d'%(target, index)

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.training_file)) and \
            os.path.exists(os.path.join(self.root, self.test_file))
    
    def generate(self, force_generate):
        if self._check_exists() and not force_generate:
            return
        
        datas = generate_dataset(train=True, n_noisy=N_noisy_train, k=K_train)
        save_pkl(datas, os.path.join(self.root, self.training_file))
        
        datas = generate_dataset(train=False, n_noisy=N_noisy_test, k=K_test)
        save_pkl(datas, os.path.join(self.root, self.test_file))
   
    def save_image(self):
        if self.train:
            image_root = os.path.join(config.data_dir, self.training_images_root)
            if os.path.exists(image_root):
                shutil.rmtree(image_root)
            save_images(zip(self.train_data, self.train_labels), image_root)
        else:
            image_root = os.path.join(config.data_dir, self.test_images_root)
            if os.path.exists(image_root):
                shutil.rmtree(image_root)
            save_images(zip(self.test_data, self.test_labels), image_root)
            
            
if __name__ == '__main__':
    dataset = MNISTNoisy(train=True, generate=True, force_generate=True)

    dataset = MNISTNoisy(train=True)
    print len(dataset)
#     mnist_multi.save_image()
    
    dataset = MNISTNoisy(train=False)
    print len(dataset)
#     dataset.save_image()
        
