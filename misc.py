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
              mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), split='train', shuffle=True, seed=None):

  #import pdb; pdb.set_trace()
  if datasetName == 'trans':
    from datasets.trans import trans as commonDataset
    import transforms.pix2pix as transforms
  elif datasetName == 'folder':
    from torchvision.datasets.folder import ImageFolder as commonDataset
    import torchvision.transforms as transforms
  elif datasetName == 'pix2pix':
    from datasets.pix2pix import pix2pix as commonDataset
    import transforms.pix2pix as transforms

  if datasetName != 'folder':
    if split == 'train':
      dataset = commonDataset(root=dataroot,
                            transform=transforms.Compose([
                              transforms.Scale(originalSize),
                              transforms.RandomCrop(imageSize),
                              transforms.RandomHorizontalFlip(),
                              transforms.ToTensor(),
                              transforms.Normalize(mean, std),
                            ]),
                            seed=seed)
    else:
      dataset = commonDataset(root=dataroot,
                            transform=transforms.Compose([
                              transforms.Scale(originalSize),
                              transforms.CenterCrop(imageSize),
                              transforms.ToTensor(),
                              transforms.Normalize(mean, std),
                             ]),
                             seed=seed)

  else:
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
                                           shuffle=shuffle, 
                                           num_workers=int(workers))
  return dataloader

def check_cuda(opt):
  if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    opt.cuda = True
  return opt


################
def accuracy(output, target, topk=(1,)):
  """Computes the precision@k for the specified values of k"""
  maxk = max(topk)
  batch_size = target.size(0)

  _, pred = output.topk(maxk, 1, True, True)
  pred = pred.t()
  correct = pred.eq(target.view(1, -1).expand_as(pred))

  res = []
  for k in topk:
    correct_k = correct[:k].view(-1).float().sum(0)
    res.append(correct_k.mul_(100.0 / batch_size))
  return res


class AverageMeter(object):
  """Computes and stores the average and current value"""
  def __init__(self):
      self.reset()

  def reset(self):
    self.val = 0
    self.avg = 0
    self.sum = 0
    self.count = 0

  def update(self, val, n=1):
    self.val = val
    self.sum += val * n
    self.count += n
    self.avg = self.sum / self.count


import numpy as np
class ImagePool:
  def __init__(self, pool_size=50):
    self.pool_size = pool_size
    if pool_size > 0:
      self.num_imgs = 0
      self.images = []

  def query(self, image):
    if self.pool_size == 0:
      return image
    if self.num_imgs < self.pool_size:
      self.images.append(image.clone())
      self.num_imgs += 1
      return image
    else:
      #import pdb; pdb.set_trace()
      if np.random.uniform(0,1) > 0.5:
        random_id = np.random.randint(self.pool_size, size=1)[0]
        tmp = self.images[random_id].clone()
        self.images[random_id] = image.clone()
        return tmp
      else:
        return image

def adjust_learning_rate(optimizer, init_lr, epoch, factor, every):
  #import pdb; pdb.set_trace()
  lrd = init_lr / every
  old_lr = optimizer.param_groups[0]['lr']
  lr = old_lr - lrd
  if lr < 0: lr = 0
  # optimizer.param_groups[0].keys()
  # ['betas', 'weight_decay', 'params', 'eps', 'lr']
  # optimizer.param_groups[1].keys()
  # *** IndexError: list index out of range
  for param_group in optimizer.param_groups:
    param_group['lr'] = lr

"""
import cv2
def visualize_attention(masks, resize, inputs=None):
  masks = masks.numpy()
  masks = masks.transpose(1,2,0)
  masks = cv2.resize(masks, (resize, resize))
  if masks.ndim == 2: masks = masks[:,:,np.newaxis]
  masks = masks.transpose(2,0,1)
  masks = torch.from_numpy(masks).unsqueeze(1)
  if inputs is not None:
    return inputs * masks.expand_as(inputs)
  else:
    return masks
"""
