# Dynamic-fixed-point-quantization
dynamic fixed point quantization method PyTorch version reffer to caffe Ristretto  

Lenet on MNIST: 99.27% ==> Lenet_8bit：99.31%; Lenet_16bit：99.22%  

VGG on CIFAR10：  

vgg_merge.py： merge conv layer and BN layer, and do quantization  

VGG：93.36% ==> VGG_8bit: 93.47%  
