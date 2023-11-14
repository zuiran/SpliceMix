import os
import json
import random
import subprocess
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

object_categories = ['airplane', 'apple', 'backpack', 'banana', 'baseball bat', 'baseball glove', 'bear', 'bed', 'bench', 'bicycle', 'bird', 'boat', 'book', 'bottle', 'bowl', 'broccoli', 'bus', 'cake', 'car', 'carrot', 'cat', 'cell phone', 'chair', 'clock', 'couch', 'cow', 'cup', 'dining table', 'dog', 'donut', 'elephant', 'fire hydrant', 'fork', 'frisbee', 'giraffe', 'hair drier', 'handbag', 'horse', 'hot dog', 'keyboard', 'kite', 'knife', 'laptop', 'microwave', 'motorcycle', 'mouse', 'orange', 'oven', 'parking meter', 'person', 'pizza', 'potted plant', 'refrigerator', 'remote', 'sandwich', 'scissors', 'sheep', 'sink', 'skateboard', 'skis', 'snowboard', 'spoon', 'sports ball', 'stop sign', 'suitcase', 'surfboard', 'teddy bear', 'tennis racket', 'tie', 'toaster', 'toilet', 'toothbrush', 'traffic light', 'train', 'truck', 'tv', 'umbrella', 'vase', 'wine glass', 'zebra']


urls = {'train_img':'http://images.cocodataset.org/zips/train2014.zip',
        'val_img' : 'http://images.cocodataset.org/zips/val2014.zip',
        'annotations':'http://images.cocodataset.org/annotations/annotations_trainval2014.zip'}

def download_coco2014(root, phase):
    work_dir = os.getcwd()
    tmpdir = os.path.join(root, 'tmp/')
    if not os.path.exists(root):
        os.makedirs(root)
    if not os.path.exists(tmpdir):
        os.makedirs(tmpdir)
    if phase == 'train':
        filename = 'train2014.zip'
    elif phase == 'val':
        filename = 'val2014.zip'
    cached_file = os.path.join(tmpdir, filename)
    if not os.path.exists(cached_file):
        print('Downloading: "{}" to {}\n'.format(urls[phase + '_img'], cached_file))
        os.chdir(tmpdir)
        subprocess.call('wget ' + urls[phase + '_img'], shell=True)
        os.chdir(root)
    # extract file
    img_data = os.path.join(root, filename.split('.')[0])
    if not os.path.exists(img_data):
        print('[dataset] Extracting tar file {file} to {path}'.format(file=cached_file, path=root))
        command = 'unzip {} -d {}'.format(cached_file,root)
        os.system(command)
    # print('[dataset] Done!')

    # train/val images/annotations
    cached_file = os.path.join(tmpdir, 'annotations_trainval2014.zip')
    if not os.path.exists(cached_file):
        print('Downloading: "{}" to {}\n'.format(urls['annotations'], cached_file))
        os.chdir(tmpdir)
        # subprocess.Popen('wget ' + urls['annotations'], shell=True)
        subprocess.call('wget ' + urls['annotations'], shell=True)
        os.chdir(root)
    annotations_data = os.path.join(root, 'annotations')
    if not os.path.exists(annotations_data):
        print('[dataset] Extracting tar file {file} to {path}'.format(file=cached_file, path=root))
        command = 'unzip {} -d {}'.format(cached_file, root)
        os.system(command)
    # print('[annotation] Done!')

    annotations_data = os.path.join(root, 'annotations')
    anno = os.path.join(root, '{}_anno.json'.format(phase))
    img_id = {}
    annotations_id = {}
    if not os.path.exists(anno):
        annotations_file = json.load(open(os.path.join(annotations_data, 'instances_{}2014.json'.format(phase))))
        annotations = annotations_file['annotations']
        category = annotations_file['categories']
        category_id = {}
        for cat in category:
            category_id[cat['id']] = cat['name']
        cat2idx = categoty_to_idx(sorted(category_id.values()))
        images = annotations_file['images']
        for annotation in annotations:
            if annotation['image_id'] not in annotations_id:
                annotations_id[annotation['image_id']] = set()
            annotations_id[annotation['image_id']].add(cat2idx[category_id[annotation['category_id']]])
        for img in images:
            if img['id'] not in annotations_id:
                continue
            if img['id'] not in img_id:
                img_id[img['id']] = {}
            img_id[img['id']]['file_name'] = img['file_name']
            img_id[img['id']]['labels'] = list(annotations_id[img['id']])
        anno_list = []
        for k, v in img_id.items():
            anno_list.append(v)
        json.dump(anno_list, open(anno, 'w'))
        if not os.path.exists(os.path.join(root, 'category.json')):
            json.dump(cat2idx, open(os.path.join(root, 'category.json'), 'w'))
        del img_id
        del anno_list
        del images
        del annotations_id
        del annotations
        del category
        del category_id
    # print('[json] Done!')
    os.chdir(work_dir)

def categoty_to_idx(category):
    cat2idx = {}
    for cat in category:
        cat2idx[cat] = len(cat2idx)
    return cat2idx

class COCO2014(Dataset):
    def __init__(self, root, transform=None, phase='train'):
        self.root = os.path.abspath(root)
        self.phase = phase
        self.img_list = []
        self.transform = transform
        download_coco2014(self.root, phase)
        self.get_anno()
        self.num_classes = len(self.cat2idx)
        print('[dataset] COCO2014 classification phase={} number of classes={}  number of images={}'.format(phase, self.num_classes, len(self.img_list)))

    def get_anno(self):
        list_path = os.path.join(self.root, '{}_anno.json'.format(self.phase))
        self.img_list = json.load(open(list_path, 'r'))
        self.cat2idx = json.load(open(os.path.join(self.root, 'category.json'), 'r'))

    def __len__(self):
        return len(self.img_list)

    def get_number_classes(self):
        return self.num_classes

    def __getitem__(self, index):
        item = self.img_list[index]
        filename = item['file_name']
        labels = sorted(item['labels'])
        target = torch.zeros(self.num_classes, dtype=torch.float32) - 1
        target[labels] = 1
        img = Image.open(os.path.join(self.root, '{}2014'.format(self.phase), filename)).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        data = {'image': img, 'name': filename, 'target': target}

        return data

    def get_number_pClasses(self):
        num_ims = self.__len__()
        npc = torch.zeros((self.get_number_classes()))
        for i in range(num_ims):
            target = self.img_list[i]['labels']
            npc[target] += 1
        dict_npc = {}
        for i, l in enumerate(object_categories):
            dict_npc[i] = npc[i]
        return npc, dict_npc

if __name__ == '__main__':

    Dset = COCO2014(root=r'Z:\pcalab\tmp\Lanke\Data_set\COCO2014', phase='val')  # val
    npc, dict_npc = Dset.get_number_pClasses()

    print(Dset[0])