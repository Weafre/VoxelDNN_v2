import contextlib
from arithmetic_coder import arithmetic_coding
import numpy as np
import os
import argparse
import time
from utils.inout import occupancy_map_explore, pc_2_block
from voxel_dnn_coder.voxel_dnn_meta_endec import save_compressed_file
import gzip
import pickle
from training.voxel_dnn_training import VoxelCNN
import tensorflow as tf

def VoxelCNN_encoding(args):
    # pc_level=9
    pc_level, ply_path,output_path,model_path128, model_path64,model_path32,model_path16,model_path8 ,bl_par_depth,signaling= args
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
    block_oc,child_blocks=pc_2_block(ply_path,pc_level,departition_level)
    _,binstr,no_oc_voxels=occupancy_map_explore(ply_path,pc_level,departition_level)
    #restore voxelCNN
    voxelCNN128 = VoxelCNN(depth=128, height=128, width=128, residual_blocks=2, n_filters=64)
    voxel_CNN128 = voxelCNN128.restore_voxelCNN(model_path128)

    voxelCNN64 = VoxelCNN(depth=64, height=64, width=64, residual_blocks=2,n_filters=64)
    voxel_CNN64 = voxelCNN64.restore_voxelCNN(model_path64)

    voxelCNN32 = VoxelCNN(depth=32, height=32, width=32, residual_blocks=2,n_filters=64)
    voxel_CNN32 = voxelCNN32.restore_voxelCNN(model_path32)

    voxelCNN16 = VoxelCNN(depth=16, height=16, width=16, residual_blocks=2,n_filters=64)
    voxel_CNN16 = voxelCNN16.restore_voxelCNN(model_path16)

    voxelCNN8 = VoxelCNN(depth=8, height=8, width=8, residual_blocks=2,n_filters=32)
    voxel_CNN8 = voxelCNN8.restore_voxelCNN(model_path8)

    voxel_CNN = [voxel_CNN128, voxel_CNN64, voxel_CNN32,voxel_CNN16,voxel_CNN8]

    #encoding blocks
    with contextlib.closing(arithmetic_coding.BitOutputStream(open(outputfile, "wb"))) as bitout, contextlib.closing(arithmetic_coding.BitOutputStream(open(outputfile+'test', "wb"))) as bitest:
        heatmap,flags,bds,no_oc_voxels=voxelCNN_encoding_slave(child_blocks,block_oc, voxel_CNN, bitout,bitest,bl_par_depth)
    with open(heatmap_file,'wb') as f:
        pickle.dump(heatmap,f)
    with gzip.open(metadata_file, "wb") as f:
        ret = save_compressed_file(binstr, pc_level, departition_level)
        f.write(ret)
    file_size = int(os.stat(outputfile).st_size) * 8
    metadata_size = int(os.stat(metadata_file).st_size) * 8 + len(flags)*2 +len(bds)*2
    avg_bpov = (file_size + metadata_size ) / no_oc_voxels
    print('Encoded file: ', ply_path)
    end = time.time()
    print('Encoding time: ', end - start)
    print('Models: ', model_path128,model_path64,model_path32 , model_path16, model_path8)
    print('Occupied Voxels: %04d' % no_oc_voxels)
    print('Blocks bitstream: ', outputfile)
    print('Metadata bitstream', metadata_file )
    print('Heatmap information: ', heatmap_file)
    print('Metadata and file size(in bits): ', metadata_size, file_size)
    print('Average bits per occupied voxels: %.04f' % avg_bpov)


def voxelCNN_encoding_slave(child_block,block_oc, voxelCNN, bitout,bitest,par_bl_level):
    static=[]
    flags=[]
    bds=[]
    enc = arithmetic_coding.ArithmeticEncoder(32, bitout)
    test_enc = arithmetic_coding.ArithmeticEncoder(32, bitest)
    no_ocv=0
    print('Encoding: ', len(child_block), 'blocks')
    for i in range(len(child_block)):
        curr_bits_cnt=[[],[],[],[]]
        box = child_block[i][0]
        ocv=np.sum(box)

        curr_level=1
        max_level=par_bl_level
        op,flag_,bd_selection=encode_child_box_test(box,child_block[i][1],block_oc,voxelCNN,test_enc,bitest,curr_level, max_level)
        fl_idx = 0
        bd_idx = 0
        _,_, curr_bits_cnt = encode_child_box_worker(box,child_block[i][1],block_oc,voxelCNN, enc, bitout, flag_, fl_idx, curr_bits_cnt, curr_level,max_level,bd_idx,bd_selection)
        for fl in flag_:
            flags.append(fl)
        for bd in bd_selection:
            bds.append(bd)
        static.append([ocv,op,flag_,curr_bits_cnt])
        no_ocv+=ocv
    enc.finish()  # Flush remaining code bits
    return static,flags,bds,no_ocv

def encode_whole_box_test(box,ref_coord, pc_oc,voxelCNN,enc,bitstream):
    extendbox_64 = [128,64]
    extendbox_32 = [128,64,32]
    extendbox_16 = [64,32,16]
    extendbox_8 = [64, 32, 16,8]
    extendbox=[extendbox_64,extendbox_32,extendbox_16,extendbox_8]
    model_index={128:0,64:1,32:2,16:3,8:4}
    pc_bbox=pc_oc.shape[1]
    box_size = box.shape[1]

    test_extend_box=extendbox[model_index[box_size]-1]
    bit=10e10
    selected_extendbox_size=len(test_extend_box)-1
    for i in range(len(test_extend_box)):
        z=ref_coord[0]
        y=ref_coord[1]
        x=ref_coord[2]
        border_size = test_extend_box[i]
        border=int((border_size-box_size)/2)
        condition=((z-border)>=0) & ((z+box_size+border)<=pc_bbox) & ((y-border)>=0) & ((y+box_size+border)<=pc_bbox) &((x-border)>=0) & ((x+box_size+border)<=pc_bbox)

        if(condition):
            Model = voxelCNN[model_index[border_size]]
            bounding_box = pc_oc[:,z-border:z+box_size+border,y-border:y+box_size+border,x-border:x+box_size+border,:]
            probs = tf.nn.softmax(Model(bounding_box)[0, :, :, :, :], axis=-1)
            probs = probs[border:border+box_size, border:border+box_size, border:border+box_size, :]
            probs = np.asarray(probs, dtype='float32')
            first_bit = bitstream.countingByte * 8
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
            curr_bit=last_bit - first_bit
            if(curr_bit<=bit):
                bit=curr_bit
                selected_extendbox_size=i
    return  bit, [selected_extendbox_size]


def encode_whole_box(box,ref_coord, pc_oc,voxelCNN,enc,bitstream, border_index):
    extendbox_64 = [128,64]
    extendbox_32 = [128,64,32]
    extendbox_16 = [64,32,16]
    extendbox_8 = [64, 32, 16,8]
    extendbox=[extendbox_64,extendbox_32,extendbox_16,extendbox_8]
    model_index={128:0,64:1,32:2,16:3,8:4}
    pc_bbox=pc_oc.shape[1]
    box_size = box.shape[1]
    if(box_size==64):
        print("encode as single block 64")
    test_extend_box = extendbox[model_index[box_size] - 1]
    border_size = test_extend_box[int(border_index)]

    z=ref_coord[0]
    y=ref_coord[1]
    x=ref_coord[2]

    border=int((border_size-box_size)/2)
    condition=((z-border)>=0) & ((z+box_size+border)<=pc_bbox) & ((y-border)>=0) & ((y+box_size+border)<=pc_bbox) &((x-border)>=0) & ((x+box_size+border)<=pc_bbox)

    if(not condition):
        print('Incorrect border size selection')
    Model = voxelCNN[model_index[border_size]]
    bounding_box = pc_oc[:,z-border:z+box_size+border,y-border:y+box_size+border,x-border:x+box_size+border,:]
    probs = tf.nn.softmax(Model(bounding_box)[0, :, :, :, :], axis=-1)
    probs = probs[border:border+box_size, border:border+box_size, border:border+box_size, :]
    probs = np.asarray(probs, dtype='float32')

    first_bit = bitstream.countingByte * 8
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

def encode_child_box_test(box,ref_coord,pc_oc,voxelCNN,test_enc,bitest,curr_level,max_level):
    child_bbox_max = int(box.shape[1]/2)
    no_bits2=0
    flag2=[]
    border_selection2=[]
    flag2.append(2)
    no_bits2+=2
    for d in range(2):
        for h in range(2):
            for w in range(2):
                child_box = box[:, d * child_bbox_max:(d + 1) * child_bbox_max,
                            h * child_bbox_max:(h + 1) * child_bbox_max,
                            w * child_bbox_max:(w + 1) * child_bbox_max, :]
                child_box_coord=[ref_coord[0]+d * child_bbox_max,ref_coord[1]+h * child_bbox_max,ref_coord[2]+w * child_bbox_max]
                child_flags=[]
                child_border_indx=[]# index from 0 to 3
                if np.sum(child_box) == 0.:
                    child_flags.append(0)
                    child_no_bits=2
                else:
                    if(curr_level>=max_level):
                        child_no_bits,border_indx=encode_whole_box_test(child_box,child_box_coord,pc_oc,voxelCNN,test_enc,bitest)
                        child_flags.append(1)
                        child_no_bits = child_no_bits+4
                        child_border_indx+=border_indx

                    else:
                        op2,rec_child_flag,child_border_selection=encode_child_box_test(child_box,child_box_coord,pc_oc, voxelCNN,test_enc,bitest,curr_level+1,max_level)
                        child_no_bits = op2
                        child_flags=rec_child_flag
                        child_border_indx=child_border_selection
                for fl in child_flags:
                    flag2.append(fl)
                border_selection2+=child_border_indx
                no_bits2+=child_no_bits
    no_bits1,border_selection1 =encode_whole_box_test(box,ref_coord,pc_oc, voxelCNN, test_enc,bitest)
    no_bits1+=2
    flag1=[1]
    if (no_bits1>no_bits2 and curr_level<=max_level):
        return no_bits2,flag2,border_selection2
    else:
        return no_bits1,flag1,border_selection1


def encode_child_box_worker(box,ref_coord,pc_oc, voxelCNN, enc, bitout,flag,fl_idx,bit_cnt,curr_level, max_level,bd_idx,border_selection):
    if flag[fl_idx]==2:
        fl_idx+=1
        child_bbox_max = int(box.shape[1] / 2)
        for d in range(2):
            for h in range(2):
                for w in range(2):
                    child_box = box[:, d * child_bbox_max:(d + 1) * child_bbox_max,
                                h * child_bbox_max:(h + 1) * child_bbox_max,
                                w * child_bbox_max:(w + 1) * child_bbox_max, :]
                    child_box_coord = [ref_coord[0] + d * child_bbox_max, ref_coord[1] + h * child_bbox_max,
                                       ref_coord[2] + w * child_bbox_max]

                    ocv=np.sum(child_box)
                    if ocv == 0.:
                        if(flag[fl_idx]!=0):
                            print('************** error: ',flag[fl_idx],fl_idx)
                            print('Level: ', curr_level)
                        fl_idx+=1
                    else:
                        if(curr_level==max_level):

                            bit_cnt[curr_level].append([ocv,encode_whole_box(child_box,child_box_coord,pc_oc, voxelCNN, enc,bitout,border_selection[bd_idx]),child_box_coord,border_selection[bd_idx]])
                            bd_idx += 1
                            if (flag[fl_idx] != 1):
                                print('************** error 1: ', flag[fl_idx], fl_idx)
                                print('Level: ', curr_level)
                            fl_idx+=1
                        else:
                            if (flag[fl_idx] == 1):
                                bit_cnt[curr_level].append([ocv,encode_whole_box(child_box,child_box_coord,pc_oc, voxelCNN, enc,bitout,border_selection[bd_idx]),child_box_coord,border_selection[bd_idx]])
                                fl_idx+=1
                                bd_idx+=1
                            elif(flag[fl_idx]==2):
                                fl_idx,bd_idx,bit_cnt=encode_child_box_worker(child_box,child_box_coord,pc_oc,voxelCNN,enc,bitout,flag,fl_idx,bit_cnt,curr_level+1,max_level,bd_idx,border_selection)
    elif flag[fl_idx]==1:
        ocv = np.sum(box)
        bit_cnt[curr_level-1].append([ocv,encode_whole_box(box,ref_coord,pc_oc, voxelCNN, enc, bitout,border_selection[bd_idx]),ref_coord,border_selection[bd_idx]])
        fl_idx+=1
        bd_idx+=1
    return fl_idx,bd_idx,bit_cnt

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
    parser.add_argument("-output", '--outputpath', type=str, help='path to output files',default='t')
    parser.add_argument("-model128", '--modelpath128', type=str, help='path to input model 64 .h5 file', default='t')
    parser.add_argument("-model64", '--modelpath64', type=str, help='path to input model 64 .h5 file',default='t')
    parser.add_argument("-model32", '--modelpath32', type=str, help='path to input model  32 .h5 file',default='t')
    parser.add_argument("-model16", '--modelpath16', type=str, help='path to input model 16 .h5 file',default='t')
    parser.add_argument("-model8", '--modelpath8', type=str, help='path to input model 8 .h5 file',default='t')
    parser.add_argument("-signaling", '--signaling', type=str, help='special character for the output',default='t')

    args = parser.parse_args()
    VoxelCNN_encoding([args.octreedepth, args.plypath,args.outputpath,args.modelpath128, args.modelpath64,args.modelpath32,args.modelpath16,args.modelpath8,args.partitioningdepth,args.signaling])
