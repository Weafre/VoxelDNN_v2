# inputs: path to saved model, path to point clouds;
# output: bit per occupied voxel

import contextlib
from arithmetic_coder import arithmetic_coding
import numpy as np
import os
import argparse
import time
from utils.inout import occupancy_map_explore_test
from voxel_dnn_coder.voxel_dnn_meta_endec import save_compressed_file,load_compressed_file
import gzip
import pickle
from training.voxel_dnn_training import VoxelCNN
import tensorflow as tf
# encoding from breadth first sequence for parallel computing
global decoding_time
decoding_time=0

def voxelCNN_decoding(args):
    pc_level, ply_path, output_path, model_path64, model_path32, model_path16, model_path8, bl_par_depth, signaling = args
    sequence_name = os.path.split(ply_path)[1]
    sequence = os.path.splitext(sequence_name)[0]

    output_path = output_path + str(sequence) + '/' + signaling
    os.makedirs(output_path, exist_ok=True)
    output_path = output_path + '/' + str(bl_par_depth) + 'levels'
    outputfile = output_path + '.blocks.bin'
    metadata_file = output_path + '.metadata.bin'
    heatmap_file = output_path + '.heatmap.pkl'
    start=time.time()
    #reading metadata
    with gzip.open(metadata_file, "rb") as f:
        decoded_binstr,pc_level, departition_level =load_compressed_file(f)
        print('First decoded depth infor: ',pc_level,departition_level)
    #getting encoding input data
    boxes,binstr,no_oc_voxels,coor_min_max,lower_level_ocv=occupancy_map_explore_test(ply_path,pc_level,departition_level)

    #restore voxelCNN
    voxelCNN64 = VoxelCNN(depth=64, height=64, width=64, residual_blocks=2, n_filters=64)
    voxel_CNN64 = voxelCNN64.restore_voxelCNN(model_path64)

    voxelCNN32 = VoxelCNN(depth=32, height=32, width=32, residual_blocks=2, n_filters=64)
    voxel_CNN32 = voxelCNN32.restore_voxelCNN(model_path32)

    voxelCNN16 = VoxelCNN(depth=16, height=16, width=16, residual_blocks=2, n_filters=64)
    voxel_CNN16 = voxelCNN16.restore_voxelCNN(model_path16)

    voxelCNN8 = VoxelCNN(depth=8, height=8, width=8, residual_blocks=2, n_filters=32)
    voxel_CNN8 = voxelCNN8.restore_voxelCNN(model_path8)

    voxel_CNN = [voxel_CNN64, voxel_CNN32, voxel_CNN16, voxel_CNN8]
    with open(heatmap_file,'rb') as f:
        heatmap=pickle.load(f)
    with open(outputfile, "rb") as inp:
        bitin = arithmetic_coding.BitInputStream(inp)
        decoded_boxes=decompress_from_adaptive_freqs(boxes,heatmap, voxel_CNN, bitin)
    end=time.time()
    global decoding_time
    decoding_time+=end-start
    decoded_boxes=decoded_boxes.astype(int)
    boxes=boxes.astype(int)
    compare=np.asarray([decoded_boxes[j] == boxes[j] for j in range(len(boxes))],dtype=int)
    print('Check 1: decoded pc level: ',pc_level)
    print('Check 2: decoded block level',  departition_level)
    print('Check 3: decoded binstr ', binstr == decoded_binstr)
    print('Check 4: decoded boxes' ,np.count_nonzero(compare),compare.all())
    print('Decoding time: ', decoding_time)

def decode_as_one(box, dec, voxelCNN):

    box_size = box.shape[1]
    idx = int(np.log2(64 / box_size))
    try:
        Model = voxelCNN[idx]
    except:
        print('index of selecting model: ', idx, 'box shape: ', box.shape)
    probs = tf.nn.softmax(Model(box)[0, :, :, :, :], axis=-1)
    probs = probs[0:box_size, 0:box_size, 0:box_size, :]
    probs = np.asarray(probs, dtype='float32')
    start = time.time()
    decoded_box=np.zeros((box_size,box_size,box_size,1))
    end = time.time()
    global  decoding_time
    decoding_time +=(end-start)*(box_size**3)
    for d in range(box_size):
        for h in range(box_size):
            for w in range(box_size):
                fre = [probs[d, h, w, 0], probs[d, h, w, 1], 0.]
                fre = np.asarray(fre)
                fre = (2 ** 10 * fre)
                fre = fre.astype(int)
                fre += 1
                freq = arithmetic_coding.NeuralFrequencyTable(fre)
                symbol = dec.read(freq)
                decoded_box[ d, h, w, 0] = symbol
    symbol = dec.read(freq)
    return dec, decoded_box

def decompress_from_adaptive_freqs(boxes,heatmap, voxelCNN, bitin):
    dec = arithmetic_coding.ArithmeticDecoder(32, bitin)
    no_box=len(boxes)
    bbox_max=boxes[0].shape[0]
    decoded_boxes=np.zeros((no_box,bbox_max,bbox_max,bbox_max,1))

    #for i in range(no_box):
    for i in range(no_box):##chang 1 to no_box if want to decode the all boxes
        print('Block ', i, '/', no_box, end='\r')
        box = []
        box.append(boxes[i, :, :, :, :])
        box = np.asarray(box)
        #print('number of non empty voxels: ', np.sum(box))
        curr_box_flag=heatmap[i][2]

        #print(' flag:', curr_box_flag)
        idx = 0
        dec,decoded_box,_=decoding_child_box_worker(box,voxelCNN,dec,curr_box_flag,idx)
        decoded_boxes[i,:,:,:,:]=decoded_box
    return decoded_boxes

def decoding_child_box_worker(box, voxelCNN, dec, flag,idx):
    box_size = box.shape[1]
    decoded_box = np.zeros(( box_size, box_size, box_size, 1))
    if flag[idx]==2:
        idx+=1
        child_bbox_max = int(box.shape[1] / 2)
        for d in range(2):
            for h in range(2):
                for w in range(2):
                    child_box=box[:, d * child_bbox_max:(d + 1) * child_bbox_max,
                                h * child_bbox_max:(h + 1) * child_bbox_max,
                                w * child_bbox_max:(w + 1) * child_bbox_max, :]
                    if(flag[idx]==0):
                        decoded_box[ d * child_bbox_max:(d + 1) * child_bbox_max,
                        h * child_bbox_max:(h + 1) * child_bbox_max,
                        w * child_bbox_max:(w + 1) * child_bbox_max, :] = 0
                        idx+=1
                    elif(flag[idx]==1):
                        dec, decoded_box[d * child_bbox_max:(d + 1) * child_bbox_max,
                        h * child_bbox_max:(h + 1) * child_bbox_max,
                        w * child_bbox_max:(w + 1) * child_bbox_max, :]=decode_as_one(child_box,dec,voxelCNN)
                        idx+=1
                    elif(flag[idx]==2):
                        dec, decoded_box[ d * child_bbox_max:(d + 1) * child_bbox_max,
                        h * child_bbox_max:(h + 1) * child_bbox_max,
                        w * child_bbox_max:(w + 1) * child_bbox_max, :],idx=decoding_child_box_worker(child_box,voxelCNN,dec,flag,idx)
    elif (flag[idx]==1):
        idx+=1
        dec,decoded_box = decode_as_one(box, dec, voxelCNN)
    return dec,decoded_box,idx


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Encoding octree')
    parser.add_argument("-level", '--octreedepth', type=int,
                        default=10,
                        help='depth of input octree to pass for encoder')
    parser.add_argument("-depth", '--partitioningdepth', type=int,
                        default=3,
                        help='max depth to partition block')
    parser.add_argument("-ply", '--plypath', type=str, help='path to input ply file')
    parser.add_argument("-output", '--outputpath', type=str, help='path to output files')
    parser.add_argument("-model64", '--modelpath64', type=str, help='path to input model 64 .h5 file')
    parser.add_argument("-model32", '--modelpath32', type=str, help='path to input model  32 .h5 file')
    parser.add_argument("-model16", '--modelpath16', type=str, help='path to input model 16 .h5 file')
    parser.add_argument("-model8", '--modelpath8', type=str, help='path to input model 8 .h5 file')
    parser.add_argument("-signaling", '--signaling', type=str, help='special character for the output')
    args = parser.parse_args()
    voxelCNN_decoding([args.octreedepth, args.plypath, args.outputpath, args.modelpath64, args.modelpath32, args.modelpath16,
         args.modelpath8,args.partitioningdepth,args.signaling])

