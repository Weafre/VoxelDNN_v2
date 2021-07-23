# inputs: path to saved model, path to point clouds;
# output: bit per occupied voxel
import contextlib
from arithmetic_coder import arithmetic_coding
import numpy as np
import os
import argparse
import time
from utils.inout import occupancy_map_explore_test
from voxel_dnn_coder.voxel_dnn_meta_endec import save_compressed_file
import gzip
import pickle
from training.voxel_dnn_training import VoxelCNN
import tensorflow as tf
print('Finished importing')
# encode using 1, 2 3,3 level
# statistic for individual block
# encoding from breadth first sequence for parallel computing
def VoxelCNN_encoding(args):
    # pc_level=9
    pc_level, ply_path,output_path, model_path64,model_path32,model_path16,model_path8 ,bl_par_depth,signaling= args
    departition_level = pc_level - 6
    sequence_name = os.path.split(ply_path)[1]
    sequence=os.path.splitext(sequence_name)[0]

    output_path = output_path+str(sequence)+'/'+signaling
    os.makedirs(output_path,exist_ok=True)
    output_path=output_path+'/'+str(bl_par_depth)+'levels'
    outputfile = output_path+'.blocks.bin'
    metadata_file = output_path + '.metadata.bin'
    heatmap_file = output_path +'.heatmap.pkl'

    start = time.time()
    #getting encoding input data
    boxes,binstr,no_oc_voxels,coor_min_max,lower_level_ocv=occupancy_map_explore_test(ply_path,pc_level,departition_level)
    #restore voxelCNN
    voxelCNN64 = VoxelCNN(depth=64, height=64, width=64, residual_blocks=2,n_filters=64)
    voxel_CNN64 = voxelCNN64.restore_voxelCNN(model_path64)

    voxelCNN32 = VoxelCNN(depth=32, height=32, width=32, residual_blocks=2,n_filters=64)
    voxel_CNN32 = voxelCNN32.restore_voxelCNN(model_path32)

    voxelCNN16 = VoxelCNN(depth=16, height=16, width=16, residual_blocks=2,n_filters=64)
    voxel_CNN16 = voxelCNN16.restore_voxelCNN(model_path16)

    voxelCNN8 = VoxelCNN(depth=8, height=8, width=8, residual_blocks=2,n_filters=32)
    voxel_CNN8 = voxelCNN8.restore_voxelCNN(model_path8)

    voxel_CNN = [voxel_CNN64, voxel_CNN32,voxel_CNN16,voxel_CNN8]

    #encoding blocks
    flags=[]
    print("Encoding: ",boxes.shape[0], ' blocks')
    with contextlib.closing(arithmetic_coding.BitOutputStream(open(outputfile, "wb"))) as bitout, contextlib.closing(arithmetic_coding.BitOutputStream(open(outputfile+'test', "wb"))) as bitest:
        heatmap,flags,no_oc_voxels=voxelCNN_encoding_slave(boxes, voxel_CNN, bitout,bitest,flags,bl_par_depth,coor_min_max,lower_level_ocv)
    with open(heatmap_file,'wb') as f:
        pickle.dump(heatmap,f)
    with gzip.open(metadata_file, "wb") as f:
        ret = save_compressed_file(binstr, pc_level, departition_level)
        f.write(ret)
    file_size = int(os.stat(outputfile).st_size) * 8
    metadata_size = int(os.stat(metadata_file).st_size) * 8 + len(flags)*2
    avg_bpov = (file_size + metadata_size ) / no_oc_voxels
    print('\n \nEncoded file: ', ply_path)
    end = time.time()
    print('Encoding time: ', end - start)
    print('Models: ',model_path64, model_path32 , model_path16, model_path8)
    print('Occupied Voxels: %04d' % no_oc_voxels)
    print('Blocks bitstream: ', outputfile)
    print('Metadata bitstream', metadata_file )
    print('Heatmap information: ', heatmap_file)
    print('Metadata and file size(in bits): ', metadata_size, file_size)
    print('Average bits per occupied voxels: %.04f' % avg_bpov)


def voxelCNN_encoding_slave(oc, voxelCNN, bitout,bitest,flags,par_bl_level,coor_min_max,lower_level_ocv):
    static=[]
    enc = arithmetic_coding.ArithmeticEncoder(32, bitout)
    test_enc = arithmetic_coding.ArithmeticEncoder(32, bitest)
    no_ocv=0
    for i in range(oc.shape[0]):
        curr_bits_cnt=[[],[],[],[]]
        #print('Block ', i , '/', oc.shape[0], end='\r')
        box = []
        box.append(oc[i, :, :, :, :])
        ocv=np.sum(oc[i])
        box = np.asarray(box)

        curr_level=1
        max_level=par_bl_level
        op,flag_=encode_child_box_test(box,voxelCNN,test_enc,bitest,curr_level, max_level)
        idx = 0
        _, curr_bits_cnt = encode_child_box_worker(box, voxelCNN, enc, bitout, flag_, idx, curr_bits_cnt, curr_level,
                                                   max_level)
        for fl in flag_:
            flags.append(fl)
        static.append([ocv,op,flag_,curr_bits_cnt,coor_min_max[i],lower_level_ocv[i]])
        #[[584.0, [1], 1018, [[[1018, 584.0]], [], [], []]]]
        #curr_bit_cnt:=[bit, ocv]
        #print(static)
        no_ocv+=ocv
    enc.finish()  # Flush remaining code bits
    return static,flags,no_ocv

def encode_whole_box(box,voxelCNN,enc,bitstream):
    #box 1x64x64x64x1 --> encode ans a box using voxel cnn
    # encoding block as one using voxelDNN
    first_bit = bitstream.countingByte * 8
    box_size=box.shape[1]
    idx=int(np.log2(64/box_size))
    try:
        Model = voxelCNN[idx]
    except:
        print('index of selecting model: ',idx, 'box shape: ', box.shape)
    probs = tf.nn.softmax(Model(box)[0, :, :, :, :], axis=-1)
    probs = probs[0:box_size, 0:box_size, 0:box_size, :]
    probs = np.asarray(probs, dtype='float32')
    for d in range(box_size):
        for h in range(box_size):
            for w in range(box_size):
                symbol = int(box[0,d, h, w, 0])
                fre = [probs[d, h, w, 0], probs[d, h, w, 1], 0.]
                fre = np.asarray(fre)
                fre = (2 ** 10 * fre)
                fre = fre.astype(int)
                fre += 1
                freq = arithmetic_coding.NeuralFrequencyTable(fre)
                enc.write(freq, symbol)
    enc.write(freq, 2)  # EOF
    last_bit = bitstream.countingByte * 8
    return last_bit-first_bit
def encode_child_box_test(box,voxelCNN,test_enc,bitest,curr_level,max_level):
    # box 1x64x64x64x1 --> flag 0 if child box 32 is empty;
    # flag 1 if non empty and encode using voxelCNN
    child_bbox_max = int(box.shape[1]/2)
    no_bits2=0
    flag2=[]
    flag2.append(2)
    no_bits2+=2
    for d in range(2):
        for h in range(2):
            for w in range(2):
                child_box = box[:, d * child_bbox_max:(d + 1) * child_bbox_max,
                            h * child_bbox_max:(h + 1) * child_bbox_max,
                            w * child_bbox_max:(w + 1) * child_bbox_max, :]
                child_flags=[]
                if np.sum(child_box) == 0.:
                    child_flags.append(0)
                    child_no_bits=2
                else:
                    #means the current block is not empty
                    if(curr_level>=max_level):
                        child_no_bits=encode_whole_box(child_box,voxelCNN,test_enc,bitest)
                        child_flags.append(1)
                        child_no_bits = child_no_bits+2
                        #print('curr level, max level, bits: ', curr_level, max_level, (last_bit - first_bit + 2))

                    else:

                        #encoding using 8 sub child blocks
                        op2,rec_child_flag=encode_child_box_test(child_box, voxelCNN,test_enc,bitest,curr_level+1,max_level)
                        child_no_bits = op2
                        child_flags=rec_child_flag
                for fl in child_flags:
                    flag2.append(fl)
                no_bits2+=child_no_bits
    no_bits1 =encode_whole_box(box, voxelCNN, test_enc,bitest)+2
    flag1=[1]
    if (no_bits1>no_bits2 and curr_level<=max_level):
        return no_bits2,flag2
    else:
        return no_bits1,flag1


def encode_child_box_worker(box, voxelCNN, enc, bitout,flag,idx,bit_cnt,curr_level, max_level):
    # box 1x64x64x64x1 --> flag 0 if child box 32 is empty;
    # flag 1 if non empty and encode using voxelCNN
    if flag[idx]==2:
        idx+=1
        child_bbox_max = int(box.shape[1] / 2)
        for d in range(2):
            for h in range(2):
                for w in range(2):
                    child_box = box[:, d * child_bbox_max:(d + 1) * child_bbox_max,
                                h * child_bbox_max:(h + 1) * child_bbox_max,
                                w * child_bbox_max:(w + 1) * child_bbox_max, :]
                    ocv=np.sum(child_box)
                    if ocv == 0.:
                        if(flag[idx]!=0):
                            print('************** causing error: ',flag[idx],idx)
                            print('Level: ', curr_level)
                        idx+=1
                    else:
                        if(curr_level==max_level):
                            bit_cnt[curr_level].append([ocv,encode_whole_box(child_box, voxelCNN, enc,bitout)])

                            if (flag[idx] != 1):
                                print('************** causing error 1: ', flag[idx], idx)
                                print('Level: ', curr_level)
                            idx+=1
                        else:
                            if (flag[idx] == 1):
                                bit_cnt[curr_level].append([ocv,encode_whole_box(child_box, voxelCNN, enc,bitout)])
                                idx+=1
                            elif(flag[idx]==2):
                            #idx+=1
                                idx,bit_cnt=encode_child_box_worker(child_box,voxelCNN,enc,bitout,flag,idx,bit_cnt,curr_level+1,max_level)
    elif flag[idx]==1:
        ocv = np.sum(box)
        bit_cnt[curr_level-1].append([ocv,encode_whole_box(box, voxelCNN, enc, bitout)])
        idx+=1
    return idx,bit_cnt


# Main launcher
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
    VoxelCNN_encoding([args.octreedepth, args.plypath,args.outputpath, args.modelpath64,args.modelpath32,args.modelpath16,args.modelpath8,args.partitioningdepth,args.signaling])
