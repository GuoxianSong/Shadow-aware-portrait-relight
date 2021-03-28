
from lib.dataset.base_dataset import BaseDataset
from PIL import Image
from torchvision.transforms import Compose, Resize, RandomCrop, CenterCrop, RandomHorizontalFlip, ToTensor, Normalize,Grayscale
import numpy as np
import os
import torch
#depth/light -> shadow map

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

class My3DDataset(BaseDataset):
    def __init__(self,opts,is_Train=True):
        self.path = opts['data_root']
        self.scene_num = opts['scene_num']
        self.subject_index_num=opts['subject_index_num']
        self.shadow_root = opts['shadow_root']
        self.light_path = opts['light_root']
        self.is_Train = is_Train
        self.train_list,self.test_list,self.train_subjects,self.train_scenes,self.test_Yb_paths,self.test_Xb_paths \
            = self.split(opts['split_files_path'])
        self.size_train_subjects = len(self.train_subjects)
        self.size_train_scenes = len(self.train_scenes)
        if(self.is_Train):
            self.size = len(self.train_list)
        else:
            self.size = len(self.test_list)

        #transforms = []
        transforms = [Resize((opts['crop_image_height'], opts['crop_image_width']), Image.BICUBIC)]
        transforms.append(ToTensor())
        transforms.append(Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
        self.transforms = Compose(transforms)

        mask_transforms = [Resize((opts['crop_image_height'], opts['crop_image_width']), Image.BICUBIC)]
        mask_transforms.append(ToTensor())
        self.mask_transforms = Compose(mask_transforms)
        self.generateMask()

        depth_transforms = []
        self.image_size = (opts['crop_image_height'], opts['crop_image_width'])
        depth_transforms.append(ToTensor())
        # depth_transforms.append(Normalize(mean=[5.9159784, 139.67622, 3.2057853], std=[33.257866, 50.370758, 35.493958]))
        self.depth_transforms = Compose(depth_transforms)
        self.generateDepth()


        light_transforms=[Resize((16, 32), Image.BICUBIC),ToTensor()]
        light_transforms.append(Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
        self.light_transforms = Compose(light_transforms)

        self.num_train = len(self.train_list)
        self.scale=2
        self.load_train_plus(opts)

    def __getitem__(self, index):
        if(self.is_Train):
            X_pattern,shadow_path,light_path,X_subject_name=self.generateFour(self.train_list[index])
            light = self.light_transforms(Image.open(light_path).convert('RGB'))
            shadow = self.transforms(Image.open(shadow_path).convert('RGB'))
            shadow = torch.mean(shadow,dim=0).unsqueeze(0)

            mask_ =self.mask_dir[X_subject_name][0:1]
            p_ = self.depth_dir[X_subject_name]
            p_ = torch.mul(p_,mask_)+(-1)*(1-mask_)
            return p_,mask_,light,shadow
        else:
            tmp = self.test_list[index].split('/')
            Xb_path = self.test_Xb_paths[index]
            b_scene = tmp[self.scene_num]
            b_scene_angle  = int(tmp[-1].split('.')[1])
            pattern = tmp[-1].split('.')[0]

            b_light_label_path = self.light_path + '%s/resize_%d.png' % (b_scene, int(b_scene_angle))
            light = self.light_transforms(Image.open(b_light_label_path).convert('RGB'))
            shadow_path = self.shadow_root + pattern + '/data/' + b_scene + '/shadow_matte/' + pattern + '.' + tmp[-1].split('.')[1]+ '.jpg'
            b_shadow = self.transforms(Image.open(shadow_path).convert('RGB'))
            b_shadow = torch.mean(b_shadow, dim=0).unsqueeze(0)
            mask_ =self.mask_dir[pattern][0:1]
            p_ = self.depth_dir[pattern]
            p_ = torch.mul(p_,mask_)+(-1)*(1-mask_)

            return p_,mask_,light,b_shadow



    # Xa,correspond 135,
    def generateFour(self,Xa_path):
        tmp = Xa_path.split('/')
        X_subject_name = tmp[self.subject_index_num]
        # Xb
        b_scene = self.train_scenes[np.random.randint(0, self.size_train_scenes)]
        b_scene_angle = '{:02}'.format(np.random.randint(1, 13))
        # shadow path
        shadow_path = self.shadow_root + X_subject_name + '/data/' + b_scene + '/shadow_matte/' + X_subject_name + '.' + b_scene_angle + '.jpg'
        X_pattern = X_subject_name
        #light path
        light_path = self.light_path+b_scene+'/resize_%d.png'%(int(b_scene_angle))

        return X_pattern,shadow_path,light_path,X_subject_name

    def __len__(self):
        return self.size