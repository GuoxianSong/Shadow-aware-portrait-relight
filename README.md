# PortraitRelight

### We public our implementation in lib.

lib/dataset: dataset files we used in training

lib/networks: networks for each pipeline

lib/models: pipelins including the loss function


### We public our rendering pipeline in render.

This needs 3D asset from: https://www.gobotree.com/cat/3d-people [detail asset can be found at gobotree_filename.txt]
Also https://www.3dscanstore.com/archviz-3d-models/female-archviz-3d-models [from 1 to 22]

lib/render/data: files about hdr and alignment vectors.

lib/render/ShadowPipeline: This can automatically generate maya file for relighting dataset.

lib/render/AutoRender: This can do batch render to generate dataset.
