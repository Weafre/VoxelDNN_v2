# Lossless Coding of Point Cloud Geometry using a Deep Generative Model
* **Authors**:
[Dat T. Nguyen](https://scholar.google.com/citations?hl=en&user=uqqqlGgAAAAJ),
[Maurice Quach](https://scholar.google.com/citations?user=atvnc2MAAAAJ),
[Giuseppe Valenzise](https://scholar.google.com/citations?user=7ftDv4gAAAAJ) and
[Pierre Duhamel](https://scholar.google.com/citations?user=gWj_W9YAAAAJ&hl=en&oi=ao)  
* **Affiliation**: Université Paris-Saclay, CNRS, CentraleSupélec, Laboratoire des signaux et systèmes, 91190 Gif-sur-Yvette, France
* **Accepted to**: [[IEEE Transactions on Circuits and Systems for Video Technology]](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=76)
* **Links**: [[Paper]](https://arxiv.org/pdf/2107.00400)
## Prerequisites
* Python 3.8
* Tensorflow 2.3.1 with CUDA 10.1.243 and cuDNN 7.6.5

Run command below to install all prerequired packages:
    
    pip3 install -r requirements.txt



## Datasets and Training

The training data are .ply files containing x,y,z coordinates of points within a d x d x d patch divided from Point Cloud. In VovxelDNN v2, we train 5 separate models for 5 block sizes (d=8,16,32,64,128). The training Point Clouds download from [ModelNet40](http://modelnet.cs.princeton.edu),[MPEG 8i](http://plenodb.jpeg.org/pc/8ilabs), [Microsoft](http://plenodb.jpeg.org/pc/microsoft), MPEG CAT1. The ModelNet40 dataset provides train and test folder separately. For MPEG and Microsoft dataset, you must manually select PCs into train and test. The commands below first select 200 densiest Point Clouds (PC) from ModelNet40, convert it from mesh to PC and then divide each PC into occupied blocks of size 64x64x64 (d=64, change the --vg_size and --level according to your block size) .

        python3 -m utils.ds_select_largest datasets/ModelNet40 datasets/ModelNet40_200 200
        python3 -m utils.ds_mesh_to_pc datasets/ModelNet40_200 datasets/ModelNet40_200_pc512 --vg_size 512
        python3 -m utils.ds_pc_octree_blocks datasets/ModelNet40_200_pc512 datasets/ModelNet40_200_pc512_oct3 --vg_size 512 --level 3 
     
      
You only need to run the last command for MPEG and Microsoft after selecting PCs into `train/` and `test/` folder:

    python  -m utils.ds_pc_octree_blocks datasets/MPEG/10bitdepth/ datasets/MPEG/10bitdepth_2_oct4/ --vg_size 1024 --level 4

The in each dataset folders, there is a original point clouds folder, and 5 folders contain ply for 5 block sizes. `datsets/` folder of MPEG and Microsoft should be like this:

    dataset/
    └── MPEG8i/
        └── 10bitdepth/              downloaded PCs from MPEG
            ├── train/               contains .ply PCs for training 
            └── test/                contains .ply PCs for validation         
        └── 10bitdepth_2_oct3/
            ├── train/               contains .ply files of 128x128x128 blocks for training 
            └── test/                contains .ply files of 128x128x128 blocks for validation
        └── 10bitdepth_2_oct4/
            ├── train/               contains .ply files of 64x64x64 blocks for training 
            └── test/                contains .ply files of 64x64x64 blocks for validation
        └── 10bitdepth_2_oct5/
            ├── train/               contains .ply files of 32x32x32 blocks for training 
            └── test/                contains .ply files of 32x32x32 blocks for validation
        └── 10bitdepth_2_oct6/
            ├── train/               contains .ply files of 16x16x16 blocks for training 
            └── test/                contains .ply files of 16x16x16 blocks for validation
        └── 10bitdepth_2_oct7/
            ├── train/               contains .ply files of 8x8x8 blocks for training 
            └── test/                contains .ply files of 8x8x8 blocks for validation
            
    └── MPEGCAT1/
        └── 10bitdepth/              downloaded PCs from MPEG
            ├── train/               contains .ply PCs for training 
            └── test/                contains .ply PCs for validation         
        └── 10bitdepth_2_oct4/
            ├── train/               contains .ply files of 64x64x64 blocks for training 
            └── test/                contains .ply files of 64x64x64 blocks for validation
        ....
            
    └── Microsoft/
        └── 10bitdepth/              downloaded PCs from Microsoft
            ├── train/               contains .ply PCs for training 
            └── test/                contains .ply PCs for validation         
        └── 10bitdepth_2_oct4/
            ├── train/               contains .ply files of 64x64x64 blocks for training 
            └── test/                contains .ply files of 64x64x64 blocks for validation
        ....


## Training
Run the following command to train block 64:
    
    python3 -m voxel_dnn_training -blocksize 64 -nfilters 64 -inputmodel Model/voxelDNN/ -outputmodel Model/voxelDNN/ -dataset datasets/ModelNet40_200_pc512_oct3/ -dataset datasets/Microsoft/10bitdepth_2_oct4/ -dataset datasets/MPEG/10bitdepth_2_oct4/  -batch 8 -epochs 50
    
## Encoder
Baseline encoding (VoxelDNN) command: 

    python3  -m  voxel_dnn_coder.voxel_dnn_abac_multi_res_sepa_model -level 10 -ply TestPC/Microsoft/10bits/phil10/ply/frame0010.ply -depth 10 -output Output/ -model64 Model/voxeldnn64/ -model32 Model/voxeldnn32/  -model16 Model/voxeldnn16/ -model8 Model/voxeldnn8/ -signaling baseline
    
    
Context extension encoder command:
    
    python3  -m  voxel_dnn_coder.voxel_dnn_extend_context2 -level 10 -ply TestPC/Microsoft_phil10_vox10_0010.ply -depth 10 -output Output/ -model128 Model/voxeldnn128/ -model64 Model/voxeldnn64/ -model32 Model/voxeldnn32/  -model16 Model/voxeldnn16/ -model8 Model/voxeldnn8/ -signaling BaselineCE
    
The encoder outputs look like this:

    Encoded file:  TestPC/Microsoft_phil10_vox10_0010.ply
    Encoding time:  7531.206112623215
    Models:  Model/voxeldnn128/ Model/voxeldnn64/ Model/voxeldnn32/ Model/voxeldnn16/ Model/voxeldnn8/
    Occupied Voxels: 1559008
    Blocks bitstream:  Output/Microsoft_phil10_vox10_0010/BaselineCE/3levels.blocks.bin
    Metadata bitstream Output/Microsoft_phil10_vox10_0010/BaselineCE/3levels.metadata.bin
    Heatmap information:  Output/Microsoft_phil10_vox10_0010/BaselineCE/3levels.heatmap.pkl
    Metadata and file size(in bits):  20404 1164544
    Average bits per occupied voxels: 0.7601

## Citation

    @article{nguyen2021lossless,
      title={Lossless Coding of Point Cloud Geometry using a Deep Generative Model},
      author={Nguyen, Dat Thanh and Quach, Maurice and Valenzise, Giuseppe and Duhamel, Pierre},
      journal={arXiv preprint arXiv:2107.00400},
      year={2021}
    }
{"mode":"full","isActive":false}