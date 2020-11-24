from torch.utils.data.dataset import Dataset
import glob
from PIL import Image
from torchvision.transforms import Compose, Resize, RandomCrop, CenterCrop, RandomHorizontalFlip, ToTensor, Normalize,Grayscale
import numpy as np
import torch
import pickle
import os
import cv2
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

class BaseDataset(Dataset):
    def __init__(self,opts,is_Train=True):
        self.path = opts['data_root']
        self.scene_num = opts['scene_num'] #scale =9
        self.subject_index_num=opts['subject_index_num'] #scale=7
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



        self.num_train = len(self.train_list)

        self.scale=2

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
        else:
            paths_ = sorted(glob.glob(self.path + "*"))
            train_list=[]
            test_list=[]
            train_subjects=[]

            _scene=[]
            for i in range(1,323):
                _scene +=[str(i)]
            train_scenes =[]
            test_scenes=[]
            #7/3 train
            for i in range(len(_scene)):
                if(i%10==1 or i%10==4 or i%10==8  ):
                    test_scenes.append('Scene'+_scene[i])
                else:
                    train_scenes.append('Scene'+_scene[i])
            with open('data/test_files.txt') as f:
                fr = f.readlines()
            patterns =[x.strip() for x in fr]
            for path_  in paths_:
                pattern  = path_.split('/')[-1]
                if(pattern in patterns):
                    subject_scenes = glob.glob(path_+'/*/*')
                    for _scene in subject_scenes:
                        if(_scene.split('/')[-1] in test_scenes):
                            test_list += glob.glob(_scene + '/*.jpg')
                else:
                    train_subjects.append(pattern)
                    subject_scenes = glob.glob(path_ + '/*/*')
                    for _scene in subject_scenes:
                        if (_scene.split('/')[-1] in train_scenes):
                            train_list += glob.glob(_scene + '/*.jpg')
            test_Yb_paths,test_Xb_paths=self.relight_pairs(test_list)


            split_file = {'train_list': train_list, 'test_list': test_list,'train_subject':train_subjects,
                          'train_scenes':train_scenes,'test_Yb_paths':test_Yb_paths,'test_Xb_paths':test_Xb_paths}
            with open(split_file_path, 'wb') as f:
                pickle.dump(split_file, f)
            return train_list,test_list,train_subjects,train_scenes,test_Yb_paths,test_Xb_paths



    def generateMask(self):
        mask_path ='Dataset/Network/Mask/*/data/albedo/*.png'
        dirs = glob.glob(mask_path)
        mask_dir={}
        for i in range(len(dirs)):
            pattern = dirs[i].split('\\')[-1].split('.')[0]
            img = self.mask_transforms(Image.open(dirs[i]).convert('RGB'))
            mask_dir[pattern]=img
        self.mask_dir=mask_dir

    def generateDepth(self):
        _path = 'Dataset/Network/Mask/*/data/Z/*.exr'
        dirs = glob.glob(_path)
        d_dir={}
        for i in range(len(dirs)):
            pattern = dirs[i].split('\\')[-1].split('.')[0]
            depth_ = cv2.imread(dirs[i],cv2.IMREAD_UNCHANGED)
            #img = cv2.cvtColor(depth_, cv2.COLOR_BGR2RGB)
            res = cv2.resize(depth_, dsize=self.image_size, interpolation=cv2.INTER_NEAREST)
            res = (res-64.5352)/48.9118
            img = self.depth_transforms(res)
            img = torch.mul(img,self.mask_dir[pattern][0:1])
            img[img>1]=0
            img[img < -1] = 0
            d_dir[pattern]=img
        self.depth_dir=d_dir


    #inpaint Yb background in test dataset
    def inpaint(self,index,path=None,mask_path =None):
        if(path is None):
            Yb_img = cv2.imread(self.test_Yb_paths[index])
            pattern = self.test_Yb_paths[index].split('/')[-1].split('.')[0]
        else:
            Yb_img = cv2.imread(path)
            pattern = path.split('/')[-1].split('.')[0]
        if(mask_path is None):
            mask_path = self.path + '..' + '/Mask/'+pattern+'/data/albedo/'+pattern+'.png'
        Yb_mask = cv2.imread(mask_path, 0)
        Yb_mask[Yb_mask != 0] = 255
        kernel = np.ones((10, 10), np.uint8)
        Yb_mask = cv2.dilate(Yb_mask, kernel, iterations=2)
        dst = cv2.inpaint(Yb_img, Yb_mask, 3, cv2.INPAINT_TELEA)

        img = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
        im_pil = Image.fromarray(img)

        transforms = [Resize((256 * self.scale, 256 * self.scale), Image.BICUBIC)]
        transforms.append(ToTensor())
        transforms.append(Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
        beatuy_transforms = Compose(transforms)

        img = beatuy_transforms(im_pil)
        return img

    def relight_pairs(self,test_list):

        Yb_paths=[]
        Xb_paths=[]
        full_subjects_num = len(self.subject_full_list)
        for i in range(len(test_list)):
            Xa_path = test_list[i]
            tmp = Xa_path.split('/')
            a_scene_angle = tmp[-1].split('.')[1]
            a_scene = tmp[self.scene_num]
            X_subject_name = tmp[self.subject_index_num]

            # random subject + scene
            b_scene = tmp[self.scene_num]
            b_scene_angle = tmp[-1].split('.')[1]
            Y_subject_name = X_subject_name
            while (b_scene == a_scene):
                b_scene = 'Scene'+str(np.random.randint(1, 323))
                b_scene_angle = '{:02}'.format(np.random.randint(1, 13))

            while (Y_subject_name == X_subject_name):
                Y_subject_name = self.subject_full_list[np.random.randint(0, full_subjects_num)]

            Yb_path = self.path + Y_subject_name + '/data/' + b_scene + '/' + Y_subject_name + '.' + b_scene_angle + '.jpg'
            Xb_path = self.path + X_subject_name + '/data/' + b_scene + '/' + X_subject_name + '.' + b_scene_angle + '.jpg'
            Yb_paths.append(Yb_path)
            Xb_paths.append(Xb_path)
        return Yb_paths,Xb_paths


    def __len__(self):
        return self.size


