import glob
import os
import os.path as osp
import sys
import torch
import torch.utils.data as data
import cv2
import numpy as np
import torchvision.transforms as T

from PIL import ImageDraw, ImageOps, Image, ImageFont
import string

tv_transform = T.Compose([
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

tv_inv_trans = T.Compose([
    T.Normalize([0, 0, 0], [1/0.229, 1/0.224, 1/0.225]),
    T.Normalize([-0.485, -0.456, -0.406], [1, 1, 1]),
    #T.ToPILImage()
    ])

class WatermarkDetection(data.Dataset):
    def __init__(self, root, transform=None, target_transform=None):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.name = 'Large-scale-Watermark'
        # anno file and img are file with same name but different postfix
        self.annos = {}
        for anno in glob.glob(osp.join(root, '*.txt')):
            fn = anno.rsplit('/', 1)[1].rsplit('.', 1)[0]
            img_fn = osp.join(root, fn+'.png')
            if osp.isfile(img_fn):
                with open(anno) as f:
                    line = f.readlines()
                    if len(line) > 1:
                        print(anno)
                    attrs = line[0][:-1].split(' ')
                self.annos[img_fn] = np.array([float(coord) for coord in attrs[2:]])[None, :]
        self.ids = list(self.annos.keys())

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img = Image.open(self.ids[idx]).convert('RGB')
        img = np.array(img)[..., ::-1]
        # img = cv2.imread(self.ids[idx])
        # if img.shape[-1] == 4:
        #     img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        #img = Image.open(self.ids[idx]).convert('RGB')
        target = self.annos[self.ids[idx]]
        height, width, _ = img.shape

        if self.target_transform is not None:
            target = self.target_transform(target, height, width)

        if self.transform is not None:
            img, boxes, _ = self.transform(img, target)
            img = img[..., ::-1]
            label_pseudo = np.ones_like(boxes)
            target = np.hstack((boxes, label_pseudo[:, 0:1]))
        img = tv_transform(img.copy().astype(np.uint8))

        return img, target

if __name__ == '__main__':
    import sys
    sys.path.append('/home/chengk/chk-root/Read/ssd.pytorch')
    from config import voc, MEANS
    from utils.augmentations import SSDAugmentation
    # from voc0712 import SSDAugmentation
    from data import *
    aug = SSDAugmentation(voc['min_dim'], MEANS)

    tmp = WatermarkDetection('/home/chengk/chk/data/Large-scale_Visible_Watermark_Dataset/watermarked_images/test', transform=aug)
    img = tmp[2]
    import pdb; pdb.set_trace()
    print(len(tmp))
    