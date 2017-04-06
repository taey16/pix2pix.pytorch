import torch
import os 
import sys

def create_exp_dir(exp):
  try:
    os.makedirs(exp)
    print('Creating exp dir: %s' % exp)
  except OSError:
    pass
  return True

def weights_init(m):
  classname = m.__class__.__name__
  if classname.find('Conv') != -1:
    m.weight.data.normal_(0.0, 0.02)
  elif classname.find('BatchNorm') != -1:
    m.weight.data.normal_(1.0, 0.02)
    m.bias.data.fill_(0)


def getLoader(datasetName, dataroot, originalSize, imageSize, batchSize=64, workers=4,
              mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), split='train'):
  #import pdb; pdb.set_trace()
  from datasets.pix2pix import pix2pix as commonDataset 
  import transforms.pix2pix as transforms
  if split == 'train':
    dataset = commonDataset(root=dataroot,
                   transform=transforms.Compose([
                     transforms.Scale(originalSize),
                     transforms.RandomCrop(imageSize),
                     transforms.RandomHorizontalFlip(),
                     transforms.ToTensor(),
                     transforms.Normalize(mean, std),
                   ]))
  else:
    dataset = commonDataset(root=dataroot,
                   transform=transforms.Compose([
                     transforms.Scale(originalSize),
                     transforms.CenterCrop(imageSize),
                     transforms.ToTensor(),
                     transforms.Normalize(mean, std),
                   ]))
  assert dataset
  dataloader = torch.utils.data.DataLoader(dataset, 
                                           batch_size=batchSize, 
                                           shuffle=True, 
                                           num_workers=int(workers))
  return dataloader
