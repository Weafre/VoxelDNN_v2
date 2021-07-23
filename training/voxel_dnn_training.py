# VoxelCNN
import random as rn
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow import nn
from tensorflow.keras import initializers
from tensorflow.keras.utils import Progbar
from utils.inout import input_fn_voxel_dnn, get_shape_data, get_files, load_points
import os
import sys
import argparse
import datetime
random_seed = 42
tf.random.set_seed(random_seed)
np.random.seed(random_seed)
rn.seed(random_seed)


# Defining main block
class MaskedConv3D(keras.layers.Layer):

    def __init__(self,
                 mask_type,
                 filters,
                 kernel_size,
                 strides=1,
                 padding='same',
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros'):
        super(MaskedConv3D, self).__init__()

        assert mask_type in {'A', 'B'}
        self.mask_type = mask_type

        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding.upper()
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)

    def build(self, input_shape):
        self.kernel = self.add_weight('kernel',
                                      shape=(self.kernel_size,
                                             self.kernel_size,
                                             self.kernel_size,
                                             int(input_shape[-1]),
                                             self.filters),
                                      initializer=self.kernel_initializer,
                                      trainable=True)

        self.bias = self.add_weight('bias',
                                    shape=(self.filters,),
                                    initializer=self.bias_initializer,
                                    trainable=True)

        center = self.kernel_size // 2

        mask = np.ones(self.kernel.shape, dtype=np.float32)
        mask[center, center, center + (self.mask_type == 'B'):, :, :] = 0.  # centre depth layer, center row
        mask[center, center + 1:, :, :, :] = 0.  # center depth layer, lower row
        mask[center + 1:, :, :, :, :] = 0.  # behind layers,all row, columns

        self.mask = tf.constant(mask, dtype=tf.float32, name='mask')

    def call(self, input):
        masked_kernel = tf.math.multiply(self.mask, self.kernel)
        x = nn.conv3d(input,
                      masked_kernel,
                      strides=[1, self.strides, self.strides, self.strides, 1],
                      padding=self.padding)
        x = nn.bias_add(x, self.bias)
        return x


class ResidualBlock(keras.Model):

    def __init__(self, h):
        super(ResidualBlock, self).__init__(name='')

        self.conv2a = keras.layers.Conv3D(filters=h, kernel_size=1, strides=1)
        self.conv2b = MaskedConv3D(mask_type='B', filters=h, kernel_size=5, strides=1)
        self.conv2c = keras.layers.Conv3D(filters=2 * h, kernel_size=1, strides=1)

    def call(self, input_tensor):
        x = nn.relu(input_tensor)
        x = self.conv2a(x)

        x = nn.relu(x)
        x = self.conv2b(x)

        x = nn.relu(x)
        x = self.conv2c(x)

        x += input_tensor
        return x


def compute_acc(y_true, y_predict,loss,writer,step):

    y_true = tf.argmax(y_true, axis=4)
    y_predict = tf.argmax(y_predict, axis=4)
    tp = tf.math.count_nonzero(y_predict * y_true, dtype=tf.float32)
    tn = tf.math.count_nonzero((y_predict - 1) * (y_true - 1), dtype=tf.float32)
    fp = tf.math.count_nonzero(y_predict * (y_true - 1), dtype=tf.float32)
    fn = tf.math.count_nonzero((y_predict - 1) * y_true, dtype=tf.float32)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    specificity = tn / (tn + fp)
    f1_score = (2 * precision * recall) / (precision + recall)
    with writer.as_default():
        tf.summary.scalar("bc/loss", loss, step)
        tf.summary.scalar("bc/precision", precision,step)
        tf.summary.scalar("bc/recall", recall,step)
        tf.summary.scalar("bc/accuracy", accuracy,step)
        tf.summary.scalar("bc/specificity", specificity,step)
        tf.summary.scalar("bc/f1_score", f1_score,step)
    a = [tp, tn, fp, fn, precision, recall, accuracy, specificity, f1_score]
    return a


class VoxelCNN():
    def __init__(self, depth=64, height=64, width=64, n_channel=1, output_channel=2,residual_blocks=2,n_filters=64):
        self.depth = depth
        self.height = height
        self.width = width
        self.n_channel = n_channel
        self.output_channel = output_channel
        self.residual_blocks=residual_blocks
        self.n_filters=n_filters
        self.init__ = super(VoxelCNN, self).__init__()

    def build_voxelCNN_model(self):
        # Creating model
        inputs = keras.layers.Input(shape=(self.depth, self.height, self.width, self.n_channel))
        x = MaskedConv3D(mask_type='A', filters=self.n_filters, kernel_size=7, strides=1)(inputs)
        for i in range(self.residual_blocks):
            x = ResidualBlock(h=int(self.n_filters/2))(x)
        x = keras.layers.Activation(activation='relu')(x)
        x = MaskedConv3D(mask_type='B', filters=self.n_filters, kernel_size=1, strides=1)(x)
        x = keras.layers.Activation(activation='relu')(x)
        x = MaskedConv3D(mask_type='B', filters=self.output_channel, kernel_size=1, strides=1)(x)
        #x = nn.softmax(x, axis=-1)#add or remove softmax here
        voxelCNN = keras.Model(inputs=inputs, outputs=x)
        #voxelCNN.summary()
        return voxelCNN


    def calling_dataset(self,training_dirs, batch_size,portion_data):
        # loading data and # Creating input stream using tf.data API
        # training_dir="../../database/ModelNet40_200_pc512_oct3_4k/**/*.ply"
        # batch_size=2
        total_files=[]
        for training_dir in training_dirs:
            training_dir=training_dir+'**/*.ply'
            p_min, p_max, dense_tensor_shape = get_shape_data(self.depth, 'channels_last')
            files = get_files(training_dir)
            total_files_len = len(files)

            # sorting and selecting files
            sizes = [os.stat(x).st_size for x in files]
            files_with_sizes = list(zip(files, sizes))
            files_sorted_by_points = sorted(files_with_sizes, key=lambda x: -x[1])
            files_sorted_by_points=files_sorted_by_points[:int(total_files_len*0.9)]
            files=list(zip(*files_sorted_by_points))
            files=list(files[0])
            files=rn.sample(files,int(len(files)*portion_data))
            total_files = np.concatenate((total_files, files), axis=0)
            print('Selected ', len(files), ' from ', total_files_len, ' in ', training_dir)

        assert len(total_files) > 0
        rn.shuffle(total_files)  # shuffle file
        print('Total blocks for training: ', len(total_files))
        points = load_points(total_files)

        files_cat = np.array([os.path.split(os.path.split(x)[0])[1] for x in total_files])
        points_train = points[files_cat == 'train']
        points_val = points[files_cat == 'test']
        number_training_data = len(points_train)
        train_dataset = input_fn_voxel_dnn(points_train, batch_size, dense_tensor_shape, 'channels_last', repeat=False,
                                 shuffle=True)
        # train_dataset=train_dataset.batch(batch_size)
        test_dataset = input_fn_voxel_dnn(points_val, batch_size, dense_tensor_shape, 'channels_last', repeat=False, shuffle=True)
        # train_dataset = input_fn_gaussain(points_train, batch_size, dense_tensor_shape, 'channels_last',gaussain_power=1, repeat=False,
        #                          shuffle=True)
        # # train_dataset=train_dataset.batch(batch_size)
        # test_dataset = input_fn_gaussain(points_val, batch_size, dense_tensor_shape, 'channels_last',gaussain_power=0, repeat=False, shuffle=True)
        return train_dataset, test_dataset,number_training_data

    def train_voxelCNN(self,batch,epochs, model_path,saved_model,dataset,portion_data):
        #log directory
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = model_path+'log' + current_time + '/train'
        test_log_dir = model_path + 'log' + current_time + '/test'
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        test_summary_writer = tf.summary.create_file_writer(test_log_dir)
        #initialize model and optimizer, loss
        voxelCNN = self.build_voxelCNN_model()
        [train_dataset, test_dataset,number_training_data] = self.calling_dataset(training_dirs=dataset, batch_size=batch,portion_data=portion_data)
        learning_rate = 1e-3
        optimizer = tf.optimizers.Adam(lr=learning_rate)
        compute_loss = keras.losses.CategoricalCrossentropy(from_logits=True, )
        n_epochs = epochs
        n_iter = int(number_training_data / batch)
        #early stopping setting
        best_val_loss, best_val_epoch = None, None
        max_patience=5
        early_stop=False
        #Load lastest checkpoint
        vars_to_load = {"Weight_biases": voxelCNN.trainable_variables, "optimizer": optimizer}
        checkpoint = tf.train.Checkpoint(**vars_to_load)
        latest_ckpt = tf.train.latest_checkpoint(saved_model)
        if latest_ckpt is not None:
           checkpoint.restore(latest_ckpt)
           print('Loaded last checkpoint')
        else:
           print('Training from scratch')
        avg_train_loss=0
        avg_test_loss=0
        ckpt_manager = tf.train.CheckpointManager(checkpoint, checkpoint_name='ckpt_', directory=model_path, max_to_keep=40)
        losses=[]
        #training
        for epoch in range(n_epochs):
            progbar = Progbar(n_iter)
            print('\n Epoch {:}/{:}'.format(epoch + 1, n_epochs))
            loss_per_epochs=[]
            for i_iter, batch_x in enumerate(train_dataset):
                batch_y = tf.cast(batch_x, tf.int32)
                with tf.GradientTape() as ae_tape:

                    logits = voxelCNN(batch_x, training=True)
                    y_true = tf.one_hot(batch_y, self.output_channel)
                    y_true = tf.reshape(y_true,(y_true.shape[0], self.depth, self.height, self.width, self.output_channel))
                    loss = compute_loss(y_true, logits)

                    metrics = compute_acc(y_true, logits,loss,train_summary_writer,int(epoch*n_iter+i_iter))
                gradients = ae_tape.gradient(loss, voxelCNN.trainable_variables)
                gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
                optimizer.apply_gradients(zip(gradients, voxelCNN.trainable_variables))
                loss_per_epochs.append(loss/batch_x.shape[0])
                progbar.add(1, values=[('loss', loss),('f1', metrics[8])])
            avg_train_loss=np.average(loss_per_epochs)
            losses.append(avg_train_loss)
            print('Avg training loss: ', avg_train_loss)

            # Validation dataset
            test_losses=[]
            test_metrics=[]
            # i=0
            for i_iter, batch_x in enumerate(test_dataset):
                batch_y = tf.cast(batch_x, tf.int32)
                logits = voxelCNN(batch_x, training=True)
                y_true = tf.one_hot(batch_y, self.output_channel)
                y_true = tf.reshape(y_true,(y_true.shape[0], self.depth, self.height, self.width, self.output_channel))

                loss = compute_loss(y_true, logits)
                metrics = compute_acc(y_true, logits,loss,test_summary_writer,i_iter)
                test_losses.append(loss/batch_x.shape[0])
                test_metrics.append(metrics)
                # i+=1
                # if(i>2000):
                #     break

            test_metrics=np.asarray(test_metrics)
            avg_metrics=np.average(test_metrics,axis=0)
            avg_test_loss=np.average(test_losses)

            #print results

            print("Testing result on epoch: %i, test loss: %f " % (epoch, avg_test_loss))
            tf.print(' tp: ', avg_metrics[0], ' tn: ', avg_metrics[1], ' fp: ', avg_metrics[2], ' fn: ', avg_metrics[3],
                     ' precision: ', avg_metrics[4], ' recall: ', avg_metrics[5], ' accuracy: ', avg_metrics[6],
                     ' specificity ', avg_metrics[7], ' f1 ', avg_metrics[8], output_stream=sys.stdout)

            if best_val_loss is None or best_val_loss > avg_test_loss:
                best_val_loss, best_val_epoch = avg_test_loss, epoch
                ckpt_manager.save()
                print('Saved model')
            if best_val_epoch < epoch - max_patience:
                print('Early stopping')
                break



    def restore_voxelCNN(self,model_path):
        voxelCNN = self.build_voxelCNN_model()
        #voxelCNN.summary()
        learning_rate = 1e-3
        optimizer = keras.optimizers.Adam(lr=learning_rate)
        vars_to_load = {"Weight_biases": voxelCNN.trainable_variables, "optimizer": optimizer}
        checkpoint = tf.train.Checkpoint(**vars_to_load)
        # Restore variables from latest checkpoint.
        latest_ckpt = tf.train.latest_checkpoint(model_path)
        if latest_ckpt is not None:
           checkpoint.restore(latest_ckpt)
        else:
           print('Can not load model: ', model_path)
        return voxelCNN
if __name__ == "__main__":
    # Command line main application function.
    parser = argparse.ArgumentParser(description='Encoding octree')
    parser.add_argument("-blocksize", '--block_size', type=int,
                        default=64,
                        help='input size of block')
    parser.add_argument("-nfilters", '--n_filters', type=int,
                        default=64,
                        help='Number of filters')
    parser.add_argument("-batch", '--batch_size', type=int,
                        default=2,
                        help='batch size')
    parser.add_argument("-epochs", '--epochs', type=int,
                        default=2,
                        help='number of training epochs')
    parser.add_argument("-inputmodel", '--savedmodel', type=str, help='path to saved model file')
    #parser.add_argument("-loss", '--loss_img_name', type=str, help='name of loss image')
    parser.add_argument("-outputmodel", '--saving_model_path', type=str, help='path to output model file')
    parser.add_argument("-dataset", '--dataset', action='append', type=str, help='path to dataset ')
    parser.add_argument("-portion_data", '--portion_data', type=float,
                        default=1,
                        help='portion of dataset to put in training, densier pc are selected first')
    args = parser.parse_args()
    block_size=args.block_size
    #batch, epochs, loss_img_path, model_path=[args.batch_size,args.epochs,args.loss_img_name,args.saving_model_path]
    voxelCNN=VoxelCNN(depth=block_size,height=block_size,width=block_size,n_channel=1,output_channel=2,residual_blocks=2,n_filters=args.n_filters)
    voxelCNN.train_voxelCNN(args.batch_size,args.epochs,args.saving_model_path,args.savedmodel, args.dataset,args.portion_data)
#python3 -m training.voxel_dnn_training -blocksize 64 -nfilters 64 -inputmodel Model/voxelDNN_CAT1 -outputmodel Model/voxelDNN_CAT1 -dataset /datnguyen_dataset/database/Modelnet40/ModelNet40_200_pc512_oct3/ -dataset /datnguyen_dataset/database/CAT1/cat1_selected_vox10_oct4/ -batch 8 -epochs 30
#python3 -m training.voxel_mixing_context -epoch 50 -blocksize 64 -outputmodel Model/voxelDnnSuperRes/ -inputmodel Model/voxelDnnSuperRes -dataset /datnguyen_dataset/database/Microsoft/10bitdepth_selected_oct4/ -dataset /datnguyen_dataset/database/MPEG/selected_8i_oct4/  -dataset /datnguyen_dataset/database/Modelnet40/ModelNet40_200_pc512_oct3/ -batch 8 -nfilters 64