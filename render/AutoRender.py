import glob
import os
from subprocess import check_output


render_path = 'D:/Autodesk/Maya2019/bin/Render.exe'
save_path = 'D:/FaceReconstruction/Dataset/NeuralRender/Images/'
ma_path ='D:/FaceReconstruction/Dataset/NeuralRender/Basic/'



class Render():
    def __init__(self,render_path,save_path,ma_path):
        self.render_path = render_path
        self.save_path =save_path
        self.ma_path =ma_path


    def render_shadow_plus(self):
        save_path_ = self.save_path + 'shadow+/'
        ma_path = self.ma_path + "shadow+/"
        render_names = glob.glob(ma_path+"*.ma")

        for i in range(len(render_names)):
            name_ = render_names[i].split('/')[-1].replace('.ma','')
            if not os.path.exists(save_path_ + name_):
                os.makedirs(save_path_ + name_)
            command = render_path + ' -proj ' + save_path_ + name_ + ' ' + ma_path + name_ + '.ma'
            try:
                check_output(command, shell=True, timeout=3600)
            except:
                print("error at ")


m=Render(render_path,save_path,ma_path)
m.render_shadow_plus()
