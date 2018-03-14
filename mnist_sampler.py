'''
Created on Oct 19, 2017

@author: longxiang
'''
from PIL import Image
import numpy as np
import config
import cPickle
from torchvision import datasets, transforms
import os

def load_pkl(path):
    f = open(path, 'rb')
    try:
        rval = cPickle.load(f)
    finally:
        f.close()
    return rval

def dump_pkl(obj, path):
    f = open(path, 'wb')
    try:
        cPickle.dump(obj, f, protocol=cPickle.HIGHEST_PROTOCOL)
    finally:
        f.close()

mnist_byte_sampler_path = os.path.join(config.data_dir, 'byte_sampler.pkl')

def load_mnist(train = True):
    train_datas = datasets.MNIST(config.data_dir, train=train, download=True,
                                 transform=transforms.Compose([
                                     transforms.Lambda(lambda x:np.array(x))
                                     ]))
    return [data for data in train_datas]

class NoisySampler(object):
    def __init__(self, cnt={}):
        self.cnt = cnt
    
    def add(self, img):
        for x in np.ndarray.flatten(img):
            self.cnt[x] = self.cnt.get(x, 0) + 1
    
    def sample(self, shape):
        a = self.cnt.keys()
        tot = sum(self.cnt.values())*1.0
        p = [self.cnt[x]/tot for x in a]
#         for x, y in zip(a, p):
#             print x, y
        np.random.choice
        img = np.random.choice(a, size=shape, p=p)
        return img
    
def get_noisy_sampler():
    if os.path.exists(mnist_byte_sampler_path):
        return load_pkl(mnist_byte_sampler_path)
    else:
        sampler = NoisySampler()
        datas = load_mnist(train = True) + load_mnist(train = False)
        for idx, (img, _) in enumerate(datas):
            if idx%1000==0:
                print idx
            sampler.add(img)
        dump_pkl(sampler, mnist_byte_sampler_path)
        return sampler
    

class NumberSampler(object):
    def __init__(self):
        self.numbers = {i:[] for i in range(10)}
    
    def add(self, img, label):
        self.numbers[label].append(img)
        
    def sample(self, label):
        return self.numbers[label][np.random.randint(len(self.numbers[label]))]
        
def get_number_sampler(datas):
    sampler = NumberSampler()
    for img, label in datas:
        sampler.add(img, label)
    return sampler
        
def to_image(img):
    return Image.fromarray(img)


def put_numbers(img, num, x=0, y=0):
    assert 0 <= x and x + num.shape[0] <= img.shape[0], 'invalid x: %d'%x
    assert 0 <= y and y + num.shape[1] <= img.shape[1], 'invalid y: %d'%y
    img[x:x+num.shape[0], y:y+num.shape[1]] = np.maximum(img[x:x+num.shape[0], y:y+num.shape[1]], num)
    
    
if __name__ == '__main__':
    img_sampler = get_noisy_sampler()
    num_sampler = get_number_sampler(load_mnist(train = True))
    
    for i in range(10):
        img =  img_sampler.sample((28,28))   
        num = num_sampler.sample(i)
        put_numbers(img, num)    
         
        img = to_image(img)
        img.save(config.data_dir + '/sample_image_%d.bmp'%i)
