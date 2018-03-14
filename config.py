'''
Created on May 9, 2017

@author: Xiang Long
'''
import os

data_dir = os.path.expanduser('~/data/MNIST')


result_dir = os.path.expanduser('~/result/video_mnist/')
 
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

model_dir = os.path.join(result_dir, 'model')
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
 
output_dir = os.path.join(result_dir, 'output')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
     
log_dir = os.path.join(result_dir, 'log')
if not os.path.exists(log_dir):
    os.makedirs(log_dir)   


def get_log_path(log_name):
    return os.path.join(log_dir, '%s.txt'%log_name)
     
def get_model_path(model_name):
    return os.path.join(model_dir, '%s.pth.tar'%model_name)
 
def get_output_path(result_name):
    return os.path.join(output_dir, '%s'%result_name)