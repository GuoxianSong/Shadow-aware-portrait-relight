
from lib.dataset.base_dataset import BaseDataset
from PIL import Image
from torchvision.transforms import Compose, Resize, RandomCrop, CenterCrop, RandomHorizontalFlip, ToTensor, Normalize,Grayscale
from PIL import ImageFile
import torch
import numpy as np
ImageFile.LOAD_TRUNCATED_IMAGES = True
#removal
class My3DDataset(BaseDataset):
    def __init__(self,opts,is_Train=True):
        self.path = opts['data_root']
        self.shadow_root = opts['shadow_root']
        self.scene_num = opts['scene_num'] #scale =9
        self.subject_index_num=opts['subject_index_num'] #scale=7
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

        #Image
        transforms = [Resize((opts['crop_image_height'], opts['crop_image_width']), Image.BICUBIC)]
        transforms.append(ToTensor())
        transforms.append(Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
        self.transforms = Compose(transforms)

        #mask
        mask_transforms = [Resize((opts['crop_image_height'], opts['crop_image_width']), Image.BICUBIC)]
        mask_transforms.append(ToTensor())
        self.mask_transforms = Compose(mask_transforms)
        self.generateMask()

        #light
        light_transforms = [Resize((256,340), Image.BICUBIC)]
        light_transforms.append(ToTensor())
        light_transforms.append(Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
        self.light_transforms = Compose(light_transforms)

        label_transforms = [Resize((16, 32), Image.BICUBIC)]
        label_transforms.append(ToTensor())
        label_transforms.append(Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
        self.label_transforms = Compose(label_transforms)


        light_weight_transforms = [Resize((16, 32), Image.BICUBIC)]
        light_weight_transforms.append(ToTensor())
        self.light_weight_transforms = Compose(light_weight_transforms)


        #depth
        depth_transforms = []
        self.image_size = (opts['crop_image_height'], opts['crop_image_width'])
        depth_transforms.append(ToTensor())
        self.depth_transforms = Compose(depth_transforms)
        self.generateDepth()
        self.load_train_plus(opts)


    def __getitem__(self, index):
        if(self.is_Train):
            Xa_path,pattern,removed_path = self.generateFour(self.train_list[index])

            #self
            Xa_out = self.transforms(Image.open(Xa_path).convert('RGB'))
            X_removal = self.transforms(Image.open(removed_path).convert('RGB'))
            mask = self.mask_dir[pattern]
            depth = self.depth_dir[pattern]

            return Xa_out, X_removal, mask, depth
        else:
            Xa_path = self.test_list[index]
            Xb_path =self.test_Xb_paths[index]
            tmp = Xb_path.split('/')
            b_scene = tmp[self.scene_num]
            removed_path = Xb_path.replace(b_scene, 'Scene135')
            pattern = tmp[-1].split('.')[0]


            mask = self.mask_dir[pattern]
            depth = self.depth_dir[pattern]

            Xa_out = self.transforms(Image.open(Xa_path).convert('RGB'))
            X_removal = self.transforms(Image.open(removed_path).convert('RGB'))


            return Xa_out, X_removal, mask, depth




    # input: source/ partial scene
    # output: relighted
    # intermediate: neuralization, depth, environment map, shadow map, light_weight map
    def generateFour(self,Xa_path):
        tmp = Xa_path.split('/')
        a_scene = tmp[-2]
        X_subject_name = tmp[self.subject_index_num]
        removed_path = Xa_path.replace(a_scene, 'Scene135')
        pattern = X_subject_name


        return Xa_path,pattern,removed_path



