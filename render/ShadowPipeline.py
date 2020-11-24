import math
import maya.cmds as cmds
import pymel.core as pm
import json
import maya.app.renderSetup.model.renderSetup as renderSetup
import math
import mtoa.utils as mutils
import maya.mel as mel
import time
import glob
index_=84

data_path = "D:\\FaceReconstruction\\Dataset\\Maya\\data\\"
save_path = "D:\\FaceReconstruction\\Dataset\\NeuralRender\\Basic\\"
obj_path = 'D:\\FaceReconstruction\\Dataset\\3D_human\\22Human\\'
hdr_path ='D:\\FaceReconstruction\\Dataset\\HDR\\HDR_haven'



class BasicModel():
    def __init__(self):
        with open(data_path+'hdr_haven.txt') as f:
            self.hdr_paths = f.read().splitlines()
        for i in range(len(self.hdr_paths)):
            path_ = self.hdr_paths[i].replace('D:\FaceReconstruction\Dataset\HDR\HDR_haven', hdr_path)
            self.hdr_paths[i] = path_

    def create_lighting(self):
        for i in range(len(self.hdr_paths)):
            self.createFileTexture(str(i + 1), self.hdr_paths[i])

    def createFileTexture(self,index_str, file_path):
        name_='file'+str(index_str)
        tex = pm.shadingNode('file', name=name_, asTexture=True, isColorManaged=True)
        if not pm.objExists('p2d_' + name_):
            pm.shadingNode('place2dTexture', name='p2d_' + name_, asUtility=True)
        p2d = pm.PyNode('p2d_' + name_)
        tex.filterType.set(0)



        pm.connectAttr(p2d.coverage, tex.coverage)
        pm.connectAttr(p2d.translateFrame, tex.translateFrame)#
        pm.connectAttr(p2d.rotateFrame, tex.rotateFrame)#
        pm.connectAttr(p2d.mirrorU, tex.mirrorU)
        pm.connectAttr(p2d.mirrorV, tex.mirrorV)

        pm.connectAttr(p2d.stagger, tex.stagger)
        pm.connectAttr(p2d.wrapU, tex.wrapU)#
        pm.connectAttr(p2d.wrapV, tex.wrapV)#
        pm.connectAttr(p2d.repeatUV, tex.repeatUV)
        pm.connectAttr(p2d.offset, tex.offset)

        pm.connectAttr(p2d.rotateUV, tex.rotateUV)
        pm.connectAttr(p2d.noiseUV, tex.noiseUV)
        pm.connectAttr(p2d.vertexUvOne, tex.vertexUvOne)
        pm.connectAttr(p2d.vertexUvTwo, tex.vertexUvTwo)
        pm.connectAttr(p2d.vertexUvThree, tex.vertexUvThree)
        pm.connectAttr(p2d.vertexCameraOne, tex.vertexCameraOne)
        pm.connectAttr(p2d.outUV, tex.uvCoord)
        pm.connectAttr(p2d.outUvFilterSize, tex.uvFilterSize)

        tex.colorSpace.set('Raw')
        tex.fileTextureName.set(file_path)
        skydome = mutils.createLocator('aiSkyDomeLight', asLight=True)
        connect_cmd ="connectAttr -force "+name_+".outColor aiSkyDomeLightShape"+index_str+".color;"
        mel.eval(connect_cmd)
        cmds.setAttr('aiSkyDomeLightShape'+index_str+'.aiSpecular',self.specular)

    def set_HDR_rotation(self):
        for i in range(12):
            for j in range(1,len(self.hdr_paths)+1):
                cmds.setKeyframe("aiSkyDomeLight" + str(j), t=i + 1, at='rotateY', v=i * 30)

    def exportFile(self,filename):
        with open(filename, "w+") as file:
            json.dump(renderSetup.instance().encode(None), fp=file, indent=2, sort_keys=True)

    def importFile(self,filename):
        with open(filename, "r") as file:
            renderSetup.instance().decode(json.load(file), renderSetup.DECODE_AND_OVERWRITE, None)

    #start from 1
    def create_layers(self):
        rs = renderSetup.instance()
        lights = []
        for i in range(len(self.hdr_paths)):
            lights.append(("aiSkyDomeLight" + str(i+1)).encode("utf-8"))

        for i in range(len(self.hdr_paths)):
            _layer = rs.createRenderLayer('Scene'+str(i))
            _layer.lightsCollectionInstance()
            _collection = _layer.createCollection("human"+str(i))
            #_collection.getSelector().staticSelection.set([human_name])
            disable_collection = _layer.createCollection("disable_light")
            tmp = list(lights)
            tmp.remove(("aiSkyDomeLight" + str(i+1)).encode("utf-8"))
            disable_collection.getSelector().staticSelection.set(tmp)
            disable_collection.setSelfEnabled(False)
        layer = rs.getDefaultRenderLayer()
        layer.setRenderable(False)

    def create(self):
        cmds.file(f=True, new=True)
        self.importFile(data_path+'render_mock_up.json')
        self.create_lighting()
        self.create_layers()
        cmds.setAttr("defaultRenderGlobals.endFrame", 9)
        cmds.file(rename=data_path+'shadow_+_model.ma')
        cmds.file(save=True, type='mayaAscii')


class ModelTool():
    def __init__(self,datafile_path=data_path+'gobotree_filename.txt'):
        with open(data_path+'hdr_haven.txt') as f:
            self.hdr_paths = f.read().splitlines()
        with open(datafile_path) as f:
            content = f.readlines()
        self.content = [x.strip() for x in content]
        self.basic_model = data_path+'shadow_+_model.ma'
        with open(data_path+'torse_vec_82.txt') as f:
            SD = f.readlines()
        self.vec = [x.strip() for x in SD]

        with open(data_path+'female_vec_22.txt') as f:
            female_SD = f.readlines()
        self.female_vec = [x.strip() for x in female_SD]



    def set_layer(self,name_):

        rs = renderSetup.instance()
        for i in range(len(self.hdr_paths)):
            _layer = rs.getRenderLayer('Scene' + str(i))
            human_collection = _layer.getCollectionByName("human"+ str(i))
            human_collection.getSelector().staticSelection.set([name_])



    def DeleteSameObject(self):
        item = ['*Hair*','*Lash*','*Wood','*Amal','*Ball','*Accounting']
        all_objects = []
        for j in range(len(item)):
            result = cmds.ls(item[j] + '*')
            result_shape = cmds.ls(item[j] + '*Shape')
            for i in range(len(result_shape)):
                result.remove(result_shape[i])
            all_objects.append(result)
        models = all_objects[0]+all_objects[1]
        for i in range(len(models)):
                cmds.delete(models[i])
        return all_objects

    def create_camera(self):
        for i in range(0,1):
            cmds.camera()
            camera_id=str(1)
            cmds.setAttr('cameraShape' + camera_id + '.focalLength', 45)
            cmds.setAttr('camera' + camera_id + '.translateX', 2)
            cmds.setAttr('camera' + camera_id + '.translateY', 159.764)
            cmds.setAttr('camera' + camera_id + '.translateZ', 71.535)
            cmds.setAttr('camera' + camera_id + '.rotateX', 0)
            cmds.setAttr('camera' + camera_id + '.rotateY',i*30)
            cmds.setAttr('camera' + camera_id + '.rotateZ', 0)
        cmds.setAttr("perspShape.renderable", False)
        for i in range(1,2):
            cmds.setAttr("cameraShape"+str(i)+".renderable", True)

    def unlocak(self,name_):
        try:
            cmds =('CBunlockAttr "OBJECT.tx"').replace('OBJECT',name_)
            mel.eval(cmds.replace('tx','tx'))
            mel.eval(cmds.replace('tx', 'ty'))
            mel.eval(cmds.replace('tx', 'tz'))
            mel.eval(cmds.replace('tx', 'rx'))
            mel.eval(cmds.replace('tx', 'ry'))
            mel.eval(cmds.replace('tx', 'rz'))
            mel.eval(cmds.replace('tx', 'tx'))
        except:
            return 0


    def set_pos(self,name_,index_):
        cmds.setAttr(name_ + '.translateX', float(self.vec[index_].split(' ')[0]))
        cmds.setAttr(name_ + '.translateY', float(self.vec[index_].split(' ')[1]))
        cmds.setAttr(name_ + '.translateZ', float(self.vec[index_].split(' ')[2]))
        cmds.setAttr(name_ + '.rotateX', float(self.vec[index_].split(' ')[3]))
        cmds.setAttr(name_ + '.rotateY', float(self.vec[index_].split(' ')[4]))
        cmds.setAttr(name_ + '.rotateZ', float(self.vec[index_].split(' ')[5]))
        cmds.setAttr(name_ + '.scaleX', float(self.vec[index_].split(' ')[6]))
        cmds.setAttr(name_ + '.scaleY', float(self.vec[index_].split(' ')[7]))
        cmds.setAttr(name_ + '.scaleZ', float(self.vec[index_].split(' ')[8]))

    def set_pos_female(self,name_,index_):
        cmds.setAttr(name_ + '.translateX', float(self.female_vec[index_].split(' ')[0]))
        cmds.setAttr(name_ + '.translateY', float(self.female_vec[index_].split(' ')[1]))
        cmds.setAttr(name_ + '.translateZ', float(self.female_vec[index_].split(' ')[2]))
        cmds.setAttr(name_ + '.rotateX', float(self.female_vec[index_].split(' ')[3]))
        cmds.setAttr(name_ + '.rotateY', float(self.female_vec[index_].split(' ')[4]))
        cmds.setAttr(name_ + '.rotateZ', float(self.female_vec[index_].split(' ')[5]))
        cmds.setAttr(name_ + '.scaleX', float(self.female_vec[index_].split(' ')[6]))
        cmds.setAttr(name_ + '.scaleY', float(self.female_vec[index_].split(' ')[7]))
        cmds.setAttr(name_ + '.scaleZ', float(self.female_vec[index_].split(' ')[8]))

    def renameFemale(self,new_name):
        result = cmds.ls('Group*')
        shape_result = cmds.ls('Group*Shape')
        result.remove(shape_result[0])
        print(result[0])
        cmds.rename(result[0],new_name)

    def set_object_rotation(self,name_):
        cmds.move(2, 0, 0, name_+".scalePivot", name_+".rotatePivot", absolute=True)
        cmds.select(name_)
        cmds.manipPivot(o=(0, 0, 0))
        origin_y_ = cmds.getAttr(name_+".ry")
        origin_x_ = cmds.getAttr(name_ + ".rx")

        for  i in range(9):
            #cmds.setKeyframe(name_, t=i + 1 , at='rotateX', v=origin_x_)
            cmds.setKeyframe(name_, t=i + 1, at='rotateY', v=origin_y_ + 10 * (i - 4))



    def create_all_female(self):
        for i in range(1,23):
            cmds.file(f=True, new=True)
            cmds.file(self.basic_model, o=True,force = True)
            name_ = 'Female_' + str(i)
            cmds.file(obj_path+'\\Female ' + str(
                i) + '\\OBJ\\Female_' + '{:02}'.format(i) + '_100k.OBJ', i=True)
            self.renameFemale('Female_' + str(i))
            self.create_camera()
            self.set_pos_female(name_, i - 1)
            #cmds.setAttr(name_ + '_BlendMaterial.reflectivity', 0)
            self.set_object_rotation(name_)
            cmds.setAttr("Female_"+str(i)+"Shape.aiSubdivType",1)
            #with open(data_path + 'good_shadow_mock_up.json', "r") as file:
            #    renderSetup.instance().decode(json.load(file), renderSetup.DECODE_AND_OVERWRITE, None)
            self.set_layer(name_)
            cmds.file(rename=save_path+'shadow+\\' + name_ + ".ma")
            cmds.file(save=True, type='mayaAscii')




    def create_all(self):
        for i in range(len(self.content)):
            cmds.file(f=True, new=True)
            cmds.file(self.basic_model, o=True,force = True)
            cmds.file(self.content[i], i=True)
            name_ = self.content[i].split('\\')[-1].split('.')[0]
            self.unlocak(name_)
            self.create_camera()
            self.DeleteSameObject()
            self.set_layer(name_)
            self.set_pos(name_, i)
            self.set_object_rotation(name_)
            cmds.setAttr(name_ + '_BlendMaterial.reflectivity', 0)
            #for j in range(0, 1):
            #    cmds.setAttr(name_ + '.rotateY', float(self.vec[i].split(' ')[4]) + j * 15)
            cmds.file(rename=save_path + 'shadow+\\' + name_ + ".ma")
            cmds.file(save=True, type='mayaAscii')

#create lighting model
basic_ = BasicModel()
basic_.create()

#align with 3D asset for two dataset
m=ModelTool()
m.create_all()
m.create_all_female()