
#light
from lib.dataset.base_dataset import BaseDataset

from PIL import Image
from torchvision.transforms import Compose, Resize, RandomCrop, CenterCrop, RandomHorizontalFlip, ToTensor, Normalize,Grayscale

import pickle
import os

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
class My3DDataset(BaseDataset):
    def __init__(self,opts,is_Train=True):
        self.path = opts['data_root']
        self.shadow_root = opts['shadow_root']
        self.scene_num = opts['scene_num'] #scale =9
        self.subject_index_num=opts['subject_index_num'] #scale=7
        self.is_Train = is_Train
        self.train_list,self.test_list,self.train_subjects,self.train_scenes,self.test_Yb_paths,self.test_Xb_paths \
            = self.split(opts['split_files_path'])

        self.test_list =self.modify_test(self.train_scenes)
        self.size_train_subjects = len(self.train_subjects)
        self.size_train_scenes = len(self.train_scenes)
        if(self.is_Train):
            self.size = len(self.train_list)
        else:
            self.size = len(self.test_list)


        #transforms = []
        transforms = [Resize((opts['crop_image_height'], opts['crop_image_width']), Image.BICUBIC)]
        transforms.append(ToTensor())
        transforms.append(Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
        self.transforms = Compose(transforms)

        label_transforms = [Resize((16, 32), Image.BICUBIC)]
        label_transforms.append(ToTensor())
        label_transforms.append(Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
        self.label_transforms = Compose(label_transforms)

        light_weight_transforms = [Resize((16, 32), Image.BICUBIC)]
        light_weight_transforms.append(ToTensor())
        self.light_weight_transforms = Compose(light_weight_transforms)

        self.num_train = len(self.train_list)
        self.light_path = self.path+'../Dataset/hdr/'
        self.load_train_plus(opts)




    def split(self,split_file_path):
        with open('data/subject_names.txt') as f:
            fr = f.readlines()
        self.subject_full_list = [x.strip() for x in fr]

        if os.path.isfile(split_file_path):
            m = pickle.load(open(split_file_path, 'rb'))
            train_list =m['train_list']
            test_list = m['test_list']
            train_subjects=m['train_subject']
            train_scenes=m['train_scenes']
            test_Yb_paths=m['test_Yb_paths']
            test_Xb_paths=m['test_Xb_paths']

            return train_list,test_list,train_subjects,train_scenes,test_Yb_paths,test_Xb_paths



    def __getitem__(self, index):
        if(self.is_Train):
            Xa_path, Xb_path=self.generateFour(self.train_list[index])
        else:
            Xa_path = self.test_list[index]
            tmp = Xa_path.split('/')
            a_scene = tmp[self.scene_num-1]
            a_angle = int(tmp[-1].split('.')[0].replace('center_',''))
            Xa_path = self.light_path + '%s/center_%d.png' % (a_scene, int(a_angle))
            Xb_path = self.test_list[index].replace('center','resize')
        beauty,label = self.GetOne(Xa_path,Xb_path)
        weight_path = (Xb_path.replace('resize','weight')).replace('hdr','hdr_weight')
        weight = self.light_weight_transforms(Image.open(weight_path).convert('RGB'))
        return beauty, label, weight

    def GetOne(self,Xa_path, Xb_path):
        beauty= self.transforms(Image.open(Xa_path).convert('RGB'))
        label = self.label_transforms(Image.open(Xb_path).convert('RGB'))
        return beauty,label


    def generateFour(self,Xa_path):
        tmp = Xa_path.split('/')

        a_scene = tmp[self.scene_num]
        a_angle = int(tmp[-1].split('.')[1])

        Xa_path = self.light_path + '%s/center_%d.png' % (a_scene, int(a_angle))
        label_path = self.light_path + '%s/resize_%d.png' % (a_scene, a_angle)
        return Xa_path,label_path

    def __len__(self):
        return self.size


