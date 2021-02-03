# PortraitRelight

### We public our implementation in lib.

lib/dataset: dataset files we used in training

lib/networks: networks for each pipeline

lib/models: pipelins including the loss function

lib/train*: training code

lib/configs: training configs

### We public our rendering pipeline in render.

This needs 3D asset from: https://www.gobotree.com/cat/3d-people [detail asset can be found at gobotree_filename.txt]
Also https://www.3dscanstore.com/archviz-3d-models/female-archviz-3d-models [from 1 to 22]
The 3D asset is not expensive, and easy to use. 

lib/render/data: files about hdr and alignment vectors.

lib/render/ShadowPipeline: This can automatically generate maya file for relighting dataset.

lib/render/AutoRender: This can do batch render to generate dataset.

Render Spec we are using: [Maya 2019](https://www.autodesk.com/products/maya) and [Arnold Render 5.3(MtoA 3.2.0)](https://www.arnoldrenderer.com/)
To accelerate the render speed, we highly recommend to enable GPU-support.
>To create maya file, please set the file paths in lib/render/ShadowPipeline.py.  
>And then in Maya command line run: 

```
python lib/render/ShadowPipeline.py
```
>After creating maya file, to do batch render the image, please run.
```
python lib/render/AutoRender.py
```


### We public our rendered images:
[dataset](https://drive.google.com/file/d/1jaN4mW-TjlSvEpO1_D15JTu7x2nO92Sv/view?usp=sharing)
(Attention, this file is about 45GB.)

Due to the current license constrain, https://www.3dscanstore.com/terms-and-conditions-licensing 
and https://www.gobotree.com/acceptable/, we might not be able directly share the rendered image and 3D asset to you. 
If you have purchased such 3D asset(inexpensive), we would like to share unzip password to you.

