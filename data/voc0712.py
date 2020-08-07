"""VOC Dataset Classes

Original author: Francisco Massa
https://github.com/fmassa/vision/blob/voc_dataset/torchvision/datasets/voc.py

Updated by: Ellis Brown, Max deGroot
"""
'''
import sys
sys.path.append('/home/chengk/chk-root/Read/ssd.pytorch')
'''
from .config import HOME
import glob
import os.path as osp
import sys
import torch
import torch.utils.data as data
import cv2
import numpy as np
from PIL import ImageDraw, ImageOps, Image, ImageFont
import string
if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

VOC_CLASSES = (  # always index 0
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor')

# note: if you used our download scripts, this should be right
VOC_ROOT = osp.join(HOME, "data/VOCdevkit/")


prints = list(string.printable)[0:84]
def random_text(img_pil):
    w, h = img_pil.size
    text_str = np.random.choice(prints, np.random.randint(low=4, high = 8))
    text_str = "".join(text_str)
    # draw the watermark on a blank
    font_size = np.random.randint(12, 50)
    font = ImageFont.truetype('/usr/share/fonts/wqy-zenhei/wqy-zenhei.ttc', font_size)
    text_width, text_height = font.getsize(text_str)
    # draw watermark on img_temp
    img_temp = Image.new('L', (int(1.2*text_width),
                                int(1.2*text_height)))
    # use a pen to draw what we want
    draw_temp = ImageDraw.Draw(img_temp) 
    opac = np.random.randint(low=255, high=256)
    draw_temp.text((0, 0), text_str,  font=font, fill=opac)
    # rotate the watermark
    rot_int = np.random.randint(low = 0, high = 8)
    rotated_text = img_temp.rotate(rot_int,  expand=1)
    '''
    '''
    col_1 = (100,100,100)
    col_2 = (np.random.randint(180, 255),
            np.random.randint(180, 255),
            np.random.randint(180, 255))
    # watermarks are drawn on the input image with white color
    '''
    col_1 = (255,255,255)
    col_2 = (255,255,255)
    '''
    #rand_loc = tuple(np.random.randint(low=0,high=max(min(h, w)-max(text_width, text_height), 1), size = (2,)))
    #print(w, text_height, h, text_height)
    rand_loc = (np.random.randint(0, max(1, w-text_width)),
                np.random.randint(0, max(1, h-text_height)))
    img_pil.paste(ImageOps.colorize(rotated_text, col_1, col_2), rand_loc,  rotated_text)
    #img_pil = Image.alpha_composite(img_pil.convert('RGBA'), ImageOps.colorize(rotated_text, col_1, col_2))
    
    # 计算watermark在img_pil的位置
    text_mask = np.array(rotated_text)
    ys, xs = text_mask.nonzero()
    x_min, x_max = xs.min(), xs.max() + 1
    y_min, y_max = ys.min(), ys.max() + 1
    
    '''
    return img_pil, (rand_loc[0]+(x_min+x_max)/2,
                    rand_loc[1]+(y_min+y_max)/2,
                    x_max-x_min, y_max-y_min, rot_int)
    '''
    return img_pil, (rand_loc[0]+x_min, rand_loc[1]+y_min, rand_loc[0]+x_max, rand_loc[1]+y_max, 1)

def scale_transform(target, height, width):
    target = list(target)
    target[0] /= width
    target[1] /= height
    target[2] /= width
    target[3] /= height
    return np.array([tuple(target)])

class VOCAnnotationTransform(object):
    """Transforms a VOC annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes

    Arguments:
        class_to_ind (dict, optional): dictionary lookup of classnames -> indexes
            (default: alphabetic indexing of VOC's 20 classes)
        keep_difficult (bool, optional): keep difficult instances or not
            (default: False)
        height (int): height
        width (int): width
    """

    def __init__(self, class_to_ind=None, keep_difficult=False):
        self.class_to_ind = class_to_ind or dict(
            zip(VOC_CLASSES, range(len(VOC_CLASSES))))
        self.keep_difficult = keep_difficult

    def __call__(self, target, width, height):
        """
        Arguments:
            target (annotation) : the target annotation to be made usable
                will be an ET.Element
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class name]
        """
        res = []
        for obj in target.iter('object'):
            difficult = int(obj.find('difficult').text) == 1
            if not self.keep_difficult and difficult:
                continue
            name = obj.find('name').text.lower().strip()
            bbox = obj.find('bndbox')

            pts = ['xmin', 'ymin', 'xmax', 'ymax']
            bndbox = []
            for i, pt in enumerate(pts):
                cur_pt = int(bbox.find(pt).text) - 1
                # scale height or width
                cur_pt = cur_pt / width if i % 2 == 0 else cur_pt / height
                bndbox.append(cur_pt)
            label_idx = self.class_to_ind[name]
            bndbox.append(label_idx)
            res += [bndbox]  # [xmin, ymin, xmax, ymax, label_ind]
            # img_id = target.find('filename').text[:-4]

        return res  # [[xmin, ymin, xmax, ymax, label_ind], ... ]


class VOCDetection(data.Dataset):
    """VOC Detection Dataset Object

    input is image, target is annotation

    Arguments:
        root (string): filepath to VOCdevkit folder.
        image_set (string): imageset to use (eg. 'train', 'val', 'test')
        transform (callable, optional): transformation to perform on the
            input image
        target_transform (callable, optional): transformation to perform on the
            target `annotation`
            (eg: take in caption string, return tensor of word indices)
        dataset_name (string, optional): which dataset to load
            (default: 'VOC2007')
    """

    def __init__(self, root,
                 image_sets=[('2007', 'trainval'), ('2012', 'trainval')],
                 transform=None, target_transform=scale_transform,
                 dataset_name='VOC0712'):
        self.root = root
        self.image_set = image_sets
        self.transform = transform
        self.target_transform = target_transform
        self.name = dataset_name
        self._annopath = osp.join('%s', 'Annotations', '%s.xml')
        self._imgpath = osp.join('%s', 'JPEGImages', '%s.jpg')
        self.ids = list()
        for (year, name) in image_sets:
            rootpath = osp.join(self.root, 'VOC' + year)
            for line in open(osp.join(rootpath, 'ImageSets', 'Main', name + '.txt')):
                self.ids.append((rootpath, line.strip()))

    def __getitem__(self, index):
        im, gt, h, w = self.pull_item(index)

        return im, gt

    def __len__(self):
        return len(self.ids)

    def pull_item(self, index):
        img_id = self.ids[index]

        #target = ET.parse(self._annopath % img_id).getroot()
        #img = cv2.imread(self._imgpath % img_id)
        img = Image.open(self._imgpath % img_id)
        img, target = random_text(img)
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        height, width, channels = img.shape

        if self.target_transform is not None:
            target = self.target_transform(target, height, width)

        if self.transform is not None:
            target = np.array(target)
            img, boxes, labels = self.transform(img, target[:, :4], target[:, 4])
            # to rgb
            img = img[:, :, (2, 1, 0)]
            # img = img.transpose(2, 0, 1)
            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))
        return torch.from_numpy(img).permute(2, 0, 1), target, height, width
        # return torch.from_numpy(img), target, height, width

    def pull_image(self, index):
        '''Returns the original image object at index in PIL form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            PIL img
        '''
        img_id = self.ids[index]
        return cv2.imread(self._imgpath % img_id, cv2.IMREAD_COLOR)

    def pull_anno(self, index):
        '''Returns the original annotation of image at index

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to get annotation of
        Return:
            list:  [img_id, [(label, bbox coords),...]]
                eg: ('001718', [('dog', (96, 13, 438, 332))])
        '''
        img_id = self.ids[index]
        anno = ET.parse(self._annopath % img_id).getroot()
        gt = self.target_transform(anno, 1, 1)
        return img_id[1], gt

    def pull_tensor(self, index):
        '''Returns the original image at an index in tensor form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            tensorized version of img, squeezed
        '''
        return torch.Tensor(self.pull_image(index)).unsqueeze_(0)

class CaptchaDetection(VOCDetection):
    def __init__(self, root, transform=None, target_transform=scale_transform):
        self.transform = transform
        self.target_transform = target_transform
        self.ids = glob.glob(osp.join(root, '*'))

    def parse_target(self, fn):
        parts = fn.rsplit('.', 1)[0].split('&')
        assert len(parts) > 4
        target = (int(p) for p in parts[-4:])
        return target

    def pull_item(self, idx):
        img_file = self.ids[idx]
        target = self.parse_target(img_file)
        img = Image.open(img_file)

        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        height, width, channels = img.shape

        if self.target_transform is not None:
            target = self.target_transform(target, height, width)
            target = np.clip(target, 0, 1)

        if self.transform is not None:
            target = np.array(target)
            img, boxes, labels = self.transform(img, target[:, :4], target[:, 4])
            # to rgb
            img = img[:, :, (2, 1, 0)]
            # img = img.transpose(2, 0, 1)
            target = np.clip(np.hstack((boxes, np.expand_dims(labels, axis=1))), 0, 1)
        return torch.from_numpy(img).permute(2, 0, 1), target, height, width

if __name__ == '__main__':
    # dataset = VOCDetection('/home/chengk/chk/VOCdevkit', target_transform=scale_transform)
    dataset = CaptchaDetection('/home/chengk/chk-root/Read/ssd.pytorch/watermark_trainset')
    img_t, target, h, w = dataset.pull_item(0)
    import pdb; pdb.set_trace()