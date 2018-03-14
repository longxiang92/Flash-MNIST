'''
Created on Oct 23, 2017

@author: longxiang
'''

import shutil
from torch.utils.data import Dataset
from mnist_sampler import *

np.random.seed(1234)
K_train = 100
K_test = 10
H = 28
W = 28
L = 25
IH = 5
IW = 5
assert L == IH*IW
B = 10
S = 2

def label_to_str(label):
    s = ''
    for i in range(B):
        if label&(1<<i):
            s+=str(i)
        else:
            s+='_'
    return s

def get_numbers(label):
    numbers = []
    for i in range(B):
        if label&(1<<i):
            numbers.append(i)
    return numbers

def generate_dataset(train=True, k=K_train, b=B, s=S):
    img_sampler = get_noisy_sampler()
    num_sampler = get_number_sampler(load_mnist(train))
    
    datas = []
    idx = 0
    for j in range(1<<b):
        for _ in range(k):
            if (idx+1)%1000 == 0 or idx+1 == k*(1<<b):
                print '%d/%d'%(idx+1, k*(1<<b))
            idx += 1
            frames = []

            numbers = get_numbers(j)
            
            for i in numbers:
                for _ in range(np.random.randint(s)+1):
                    img = img_sampler.sample((H,W))
                    num = num_sampler.sample(i)
                    put_numbers(img, num)
                    frames.append(img)
            
            for i in range(L-len(frames)):
                img = img_sampler.sample((H,W))
                frames.append(img)
            
            np.random.shuffle(frames)
            video = np.array(frames)
            datas.append((video, j))
    return datas

def save_pkl(datas, path):
    print 'save pkl to: %s'%path
    train_imgs, train_labels = zip(*datas)
    dump_pkl((train_imgs, train_labels), path)
    

def save_images(datas, image_dir):
    print 'save videos to: %s' %image_dir
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
    idx = 0
    for video, label in datas:
        img = np.zeros((IH*H,IW*W), dtype='uint8')
        for i, frame in enumerate(video):
            x = i/IH
            y = i%IH
            img[x*H:(x+1)*H, y*W:(y+1)*W] = frame
        img = to_image(img)
        img.save(os.path.join(image_dir, '%s %06d.bmp'%(label_to_str(label), idx)))
        idx += 1
        

class MNISTFlash(Dataset):
    training_images_root = 'flash_train'
    test_images_root = 'flash_test'
    training_file = 'flash_train.pkl'
    test_file = 'flash_test.pkl'

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
            video, target = self.train_data[index], self.train_labels[index]
        else:
            video, target = self.test_data[index], self.test_labels[index]

        imgs = [Image.fromarray(img, mode='L') for img in video]

        if self.transform is not None:
            imgs = self.transform(imgs)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return imgs, target, '%s %06d'%(label_to_str(target), index)

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def _check_exists(self):
#         return True
        return os.path.exists(os.path.join(self.root, self.training_file)) and \
            os.path.exists(os.path.join(self.root, self.test_file))
    
    def generate(self, force_generate):
        if self._check_exists() and not force_generate:
            return
        
        datas = generate_dataset(train=True, k=K_train)
        save_pkl(datas, os.path.join(self.root, self.training_file))
        
        datas = generate_dataset(train=False, k=K_test)
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
    dataset = MNISTFlash(train=False, generate=True, force_generate=True)

#     dataset = MNISTFlash(train=True)
#     print len(dataset)
#     dataset.save_image()
     
#     dataset = MNISTFlash(train=False)
#     print len(dataset)
#     dataset.save_image()
