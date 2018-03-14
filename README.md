# Flash-MNIST
Toy dataset in "Attention Clusters: Purely Attention Based Local Feature Integration for Video Classification"

Attention Clusters: Purely Attention Based Local Feature Integration for Video Classification. Xiang Long, Chuang Gan, Gerard de Melo, Jiajun Wu, Xiao Liu, Shilei Wen. CVPR 2018.


## Requirements
Anaconda2

Pytorch


## Pretrain CNN model
```
python main_mnist_noisy.py 
```

## Extract local features
```
python main_get_feature.py 
```

## Training and Evaluation

```
python main_feature_satt.py --natt=N
```


