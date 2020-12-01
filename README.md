# PortraitRelight

### We public our implementation in lib.

lib/dataset: dataset files we used in training

lib/networks: networks for each pipeline

lib/models: pipelins including the loss function


### We public our rendering pipeline in render.

This needs 3D asset from: https://www.gobotree.com/cat/3d-people [detail asset can be found at gobotree_filename.txt]
Also https://www.3dscanstore.com/archviz-3d-models/female-archviz-3d-models [from 1 to 22]
Since we cannot directly share the 3D asset, we high recommend those who are interesed in this dataset to purchase the 3D asset. 

lib/render/data: files about hdr and alignment vectors.

lib/render/ShadowPipeline: This can automatically generate maya file for relighting dataset.

lib/render/AutoRender: This can do batch render to generate dataset.

### We public our rendered images:
(attention, this file is about 45GB)
Rendered Image: https://drive.google.com/file/d/1jaN4mW-TjlSvEpO1_D15JTu7x2nO92Sv/view?usp=sharing
Please note that this dataset is only for research purpose. Since those images is rendered using 3D asset,
you also need to agree the following terms: https://www.3dscanstore.com/terms-and-conditions-licensing 
and https://www.gobotree.com/acceptable/.
Please send us(gxsong.ntu@gmail.com) your agreement, we will reply the password to unzip dataset. 
