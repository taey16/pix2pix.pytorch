# [Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/abs/1611.07004)
# Install
- install **fantastic** [pytorch](https://github.com/pytorch/pytorch) and [pytorch.vision](https://github.com/pytorch/vision)

# Datasets
- [Download images from author's implementation](https://github.com/phillipi/pix2pix)
- Suppose you downloaded the "facades" dataset in /path/to/facades

# Train with facades dataset (mode: B2A)
- ```CUDA_VISIBLE_DEVICES=x python main_pix2pixgan.py --dataset pix2pix --dataroot /path/to/facades/train --valDataroot /path/to/facades/val --mode B2A --exp ./facades --display 5 --evalIter 500```
 - Resulting model is saved in ./facades directory named like net[D|G]_epoch_xx.pth
# Train with edges2shoes dataset (mode: A2B)
- ```CUDA_VISIBLE_DEVICES=x python main_pix2pixgan.py --dataset pix2pix --dataroot /path/to/edges2shoes/train --valDataroot /path/to/edges2shoes/val --mode A2B --exp ./edges2shoes --batchSize 4 --display 5```

# Results
- Randomly selected input samples
![input](https://github.com/taey16/pix2pix.pytorch/blob/master/imgs/real_input.png)
- Corresponding real target samples
![target](https://github.com/taey16/pix2pix.pytorch/blob/master/imgs/real_target.png)
- **Corresponding generated samples**
![generated](https://github.com/taey16/pix2pix.pytorch/blob/master/imgs/generated_epoch_00000212_iter00085000.png)

# Note
- We modified [pytorch.vision.folder](https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py) and [transform.py](https://github.com/pytorch/vision/blob/master/torchvision/transforms.py) as to follow the format of train images in the datasets
- Most of the parameters are the same as the paper.
- You can easily reproduce results of the paper with other dataets
- Try B2A or A2B translation as your need

# Reference
- [pix2pix.torch](https://github.com/phillipi/pix2pix)
- [pix2pix-pytorch](https://github.com/mrzhu-cool/pix2pix-pytorch) (Another pytorch implemention of the pix2pix)
- [dcgan.pytorch](https://github.com/pytorch/examples/tree/master/dcgan)
- **FANTASTIC pytorch** [pytorch doc](http://pytorch.org/docs/notes/autograd.html)
- [genhacks from soumith](https://github.com/soumith/ganhacks)
