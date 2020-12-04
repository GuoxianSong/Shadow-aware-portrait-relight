from lib.utils.utils import test_write_images,get_config,prepare_sub_folder
from torch.utils.data import DataLoader
from lib.models.end2end_model import Models
from lib.dataset.end2end_dataset import My3DDataset
import torch.backends.cudnn as cudnn
import os
import torch
import time
import numpy as np

class RTenser():
    def __init__(self):
        cudnn.benchmark = True
        # Load experiment setting
        self.name_ = 'end2end'
        config = get_config('configs/%s.yaml'%(self.name_))
        self.trainer = Models(config)
        self.trainer.cuda()

        # Setup logger and output folders
        self.trainer.resume('outputs/%s/checkpoints'%(self.name_), hyperparameters=config,need_opt=False,path='outputs/end2end/checkpoints/gen_00115000.pt')
        self.trainer.eval()
        self.config = config
        self.dataset = My3DDataset(opts=self.config, is_Train=False)
        self.test_loader = DataLoader(dataset=self.dataset, batch_size=int(self.config['batch_size']*5), shuffle=False,
                                      num_workers=self.config['nThreads'])

    def upload(self,img, mask):
        img = torch.unsqueeze(img, dim=0)
        mask = torch.unsqueeze(mask, dim=0)
        img = img.cuda(self.config['cuda_device']).detach()
        mask = mask.cuda(self.config['cuda_device']).detach()
        return img, mask


    def test_real(self):
        from PIL import Image
        import cv2
        self.dataset.scale = 2
        self.dataset.load_real()

        def adjust_gamma(image, gamma=2.0):
            invGamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** invGamma) * 255
                              for i in np.arange(0, 256)]).astype("uint8")
            return cv2.LUT(image, table)
        line =  cv2.resize(cv2.imread(self.dataset.light_path + '../preview/mask_line.png'),(1024,512))/255
        region = cv2.resize(cv2.imread(self.dataset.light_path + '../preview/mask_r.png'),(1024,512))/255


        if(not os.path.exists('outputs/%s/images/test_real'%(self.name_))):
            os.mkdir('outputs/%s/images/test_real'%(self.name_))
        tmp = np.loadtxt('data/test_plus_scene.txt').astype(int)
        with torch.no_grad():
            source_img, source_mask = self.dataset.getReal_path(r'D:\Research\Shadow\rebuttal\examples\512\%s' % '041785')
            source_img, source_mask = self.upload(source_img, source_mask)
            for i in range(len(tmp)):
                b_scene =tmp[i]
                b_scene_angle = 1
                light_path = self.dataset.light_path + 'Scene%s/center_%d.png' % (b_scene, int(b_scene_angle))
                light = self.dataset.light_transforms(Image.open(light_path).convert('RGB'))
                light = torch.unsqueeze(light, dim=0).cuda(self.config['cuda_device']).detach()

                out,out_remove,out_depth,out_light,out_shadow = self.trainer.forward_test_real(source_img, source_mask, light)

                test_write_images(out, 1, 'tmp/tmp_source.png')

                test_write_images(2 * (source_mask[0:1].detach().cpu()[:3] - 0.5), 1, 'tmp/tmp_source_mask.png')
                dim = (512, 512)
                bg = cv2.imread(self.dataset.light_path + 'Scene%s/hdr.%s.jpg' % (b_scene,'{:02}'.format(b_scene_angle)))
                human = cv2.imread('tmp/tmp_source.png')

                mask = cv2.imread('tmp/tmp_source_mask.png')
                mask = mask / 256
                composied_x = int(len(bg) - dim[0])
                composied_y = int(len(bg[0]) / 2 - dim[1] / 2)
                bg[composied_x:composied_x + dim[0], composied_y:composied_y + dim[1]] = \
                    bg[composied_x:composied_x + dim[0], composied_y:composied_y + dim[1]] * (1 - mask) + human * mask
                cv2.imwrite('outputs/%s/images/test_real/%d.png' % (self.name_, i), bg)
                if(0):
                    hdr = adjust_gamma(cv2.imread(self.dataset.light_path + '../preview/%d.jpg' % b_scene))
                    hdr = np.multiply(0.4 * hdr, 1 - region) + np.multiply(hdr, region)
                    hdr = np.multiply(hdr, 1 - line) + np.multiply(255, line)

                    # hdr = np.swapaxes(hdr,0,1)
                    cv2.imwrite('outputs/%s/images/test_real/%d_hdr.jpg' % (self.name_, i), hdr)
                    test_write_images(out_light, 1, 'outputs/%s/images/test_real/%d_hdr_light.jpg' % (self.name_, i))


    def consist_check(self):
        import shutil
        import cv2

        self.dataset.scale=2
        with open('data/test_files.txt') as f:
            fr = f.readlines()
        test_subject = [x.strip() for x in fr]
        Scene = [311, 286, 236, 242, 3, 286, 190]
        if(0):
            subject = test_subject[0]
            print(subject)
            with torch.no_grad():
                for subject in test_subject:
                    for i in Scene:
                        source_img, source_mask = self.dataset.getReal_path('consist/%s/%d' %(subject,i),'consist/%s/%s.png'%(subject,subject))
                        source_img, source_mask = self.upload(source_img, source_mask)
                        removal_x, depth_x = self.trainer.removal_gen.forward(torch.mul(source_img, source_mask))
                        removal_x = torch.mul(removal_x,source_mask)+(-1)*(1-source_mask)
                        test_write_images(removal_x, 1, 'outputs/end2end/images/consist/%s_%d.jpg' % (subject,i))
        else:
            subject = 'GTP_CMan_Filip_Smr_09_Wk_Adl_Ccs_Adl_Ccs_Mgr'
            img = np.zeros((5,512,512))
            Scene = [311,242,236,286,190]
            for i in range(len(Scene)):
                img[i] = cv2.imread('outputs/end2end/images/consist/%s_%d.jpg' % (subject,Scene[i]),0)/255.0
            out = img.std(axis=0)
            import seaborn as sns;
            sns.set_theme()

            ax = sns.heatmap(out, vmin=0, vmax=0.1,yticklabels=False,xticklabels=False)
            ax.figure.savefig("consist.png")

    def step_by_step(self):
        from PIL import Image
        import torchvision.transforms.functional as TF
        import cv2
        self.dataset.scale = 2
        self.dataset.load_real()
        if(not os.path.exists('outputs/%s/images/step'%(self.name_))):
            os.mkdir('outputs/%s/images/step'%(self.name_))
        tmp = np.loadtxt('data/test_plus_scene.txt').astype(int)
        with torch.no_grad():
            source_img, source_mask = self.dataset.getReal_path(r'D:\Research\Shadow\rebuttal\examples\512\%s' % 'Pants_00894')
            source_img, source_mask = self.upload(source_img, source_mask)
            for i in range(len(tmp)):
                b_scene =tmp[i]
                b_scene_angle = 1
                light_path = self.dataset.light_path + 'Scene%s/center_%d.png' % (b_scene, int(b_scene_angle))
                light = self.dataset.light_transforms(Image.open(light_path).convert('RGB'))
                light = torch.unsqueeze(light, dim=0).cuda(self.config['cuda_device']).detach()

                out,out_remove,out_depth,out_light,out_shadow = self.trainer.forward_test_real(source_img, source_mask, light)
                out_shadow = torch.mul(out_shadow,source_mask.cpu())+(-1)*(1-source_mask.cpu())
                out_depth = torch.clamp(torch.mul( out_depth*3.2,source_mask.cpu())+(-1)*(1-source_mask.cpu()),-1,1)

                bg = Image.open(self.dataset.light_path + 'Scene%s/hdr.%s.jpg' % (b_scene,'{:02}'.format(b_scene_angle)))
                bg = TF.to_tensor(bg).unsqueeze_(0)[:,:,208:, 384:896]
                bg=(bg-0.5)*2
                out = torch.mul(out, source_mask.cpu()) + (bg) * (1 - source_mask.cpu())
                test_write_images(out, 1,'outputs/%s/images/step/%d_0.png' % (self.name_, i))
                test_write_images(out_remove, 1, 'outputs/%s/images/step/%d_1.png' % (self.name_, i))
                test_write_images(out_depth, 1, 'outputs/%s/images/step/%d_2.png' % (self.name_, i))
                test_write_images(out_shadow, 1, 'outputs/%s/images/step/%d_3.png' % (self.name_, i))





R = RTenser()
R.test_real()