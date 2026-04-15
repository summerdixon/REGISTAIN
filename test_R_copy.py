import os

import scipy.io

import ops

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import glob
from configobj import ConfigObj
from models import att_unet_2d
from tqdm import tqdm
from losses import *
import scipy.io as sio
import matplotlib.pyplot as plt
import batch_utils
from batch_utils import ImageTransformationBatchLoader_Testing
from models.aligners import aligner_unet_cvpr2018, aligner_unet_cvpr2018_vJX
import time
from scipy.io import savemat

def init_parameters():
    tc, vc = ConfigObj(), ConfigObj()
    tc.image_path = 'L:\\Pneumonia_Dataset\\Second_reg\\Testing\\AAW-19-00033-A19_good\\target\\*.mat'
    vc.image_path = 'L:\\Pneumonia_Dataset\\Second_reg\\Testing\\AAW-19-00033-A19_good\\target\\*.mat'
    # tc.image_path = 'I:\\Pneumonia_Dataset\\Second_reg\\Testing\\AAW-18-00044-A6_wait\\target\\*.mat'
    # vc.image_path = 'I:\\Pneumonia_Dataset\\Second_reg\\Testing\\AAW-18-00044-A6_wait\\target\\*.mat'

    def convert_inp_path_from_target(inp_path: str):
        return inp_path.replace('target', 'input')
    # def convert_inp_path_from_target(inp_path: str):
    #     return inp_path.replace('/target', '/input')

    tc.convert_inp_path_from_target = convert_inp_path_from_target
    vc.convert_inp_path_from_target = convert_inp_path_from_target

    tc.is_mat, vc.is_mat = True, True  # True for .mat, False for .npy
    tc.data_inpnorm, vc.data_inpnorm = False, False  # True for normalizing input images

    tc.channel_start_index, vc.channel_start_index = 0, 0
    tc.channel_end_index, vc.channel_end_index = 2, 2  # exclusive

    tc.is_training, vc.is_training = True, False
    tc.image_size, vc.image_size = 2048, 2048   # 1408, 1408
    tc.num_slices, vc.num_slices = 2, 2
    tc.label_channels, vc.label_channels = 3, 3

    tc.nf_enc, vc.nf_enc = [8, 16, 16, 32, 32], [8, 16, 16, 32, 32]  # for aligner
    tc.nf_dec, vc.nf_dec = [32, 32, 32, 32, 32, 16, 16], [32, 32, 32, 32, 32, 16, 16]  # for aligner
    tc.R_loss_type = 'ncc'
    tc.lambda_r_tv = 1.0  # .1    # tv of predicted flow
    tc.gauss_kernal_size = 80
    tc.dvf_clipping = True  # clip DVF to [mu-sigma*dvf_clipping_nsigma, mu+sigma*dvf_clipping_nsigma]
    tc.dvf_clipping_nsigma = 3
    tc.dvf_thresholding = True  # clip DVF to [-dvf_thresholding_distance, dvf_thresholding_distance]
    tc.dvf_thresholding_distance = 50
    tc.loss_mask, vc.loss_mask = False, False  # True, False

    assert tc.channel_end_index - tc.channel_start_index == tc.num_slices
    assert vc.channel_end_index - vc.channel_start_index == vc.num_slices
    tc.n_channels, vc.n_channels = 32, 32

    tc.batch_size, vc.batch_size = 1, 1
    tc.n_threads, vc.n_threads = 2, 2
    tc.q_limit, vc.q_limit = 10, 10
    tc.n_shuffle_epoch, vc.n_shuffle_epoch = 1, 1  # for the batchloader
    tc.data_inpnorm, vc.data_inpnorm = 'norm_by_mean_std', 'norm_by_mean_std'

    return tc, vc


if __name__ == '__main__':

    model_path = 'M:/Regstain_Code/code/stage2_20220727_G&RSeperateTrain_initIter=0'
    checkpoint_path_G = model_path + '/model_G_iter=87700.h5'
    checkpoint_path_R = model_path + '/model_R_iter=87700.h5'

    output_path = 'Y:/Cycle_consistency/Autopsy_Testing_Dataset_Registered/AAW-19-00033-A19_good'
    tf.io.gfile.mkdir(output_path)
    tf.io.gfile.mkdir(output_path + '/target_registered/')

    # initialize architecture and load weights
    tc, vc = init_parameters()

    model_G = att_unet_2d((tc.image_size, tc.image_size, tc.num_slices), n_labels=tc.label_channels, name='model_G',
                          filter_num=[tc.n_channels, tc.n_channels * 2, tc.n_channels * 4, tc.n_channels * 8,
                                      tc.n_channels * 16],
                          stack_num_down=3, stack_num_up=3, activation='LeakyReLU',
                          atten_activation='ReLU', attention='add',
                          output_activation=None, batch_norm=True, pool='ave', unpool='bilinear')
    model_G.load_weights(checkpoint_path_G)

    model_R = aligner_unet_cvpr2018_vJX([tc.image_size, tc.image_size], tc.nf_enc, tc.nf_dec,
                                        gauss_kernal_size=tc.gauss_kernal_size,
                                        flow_clipping=tc.dvf_clipping,
                                        flow_clipping_nsigma=tc.dvf_clipping_nsigma,
                                        flow_thresholding=tc.dvf_thresholding,
                                        flow_thresh_dis=tc.dvf_thresholding_distance,
                                        loss_mask=tc.loss_mask, loss_mask_from_prev_cascade=False)
    model_R.load_weights(checkpoint_path_R)

    # _, _, test_images = batch_utils.Her2data_splitter(tc)
    test_images = glob.glob(vc.image_path)

    valid_bl = ImageTransformationBatchLoader_Testing(test_images, vc, vc.num_slices, is_testing=True,
                                              n_parallel_calls=vc.n_threads, q_limit=vc.q_limit,
                                              n_epoch=vc.n_shuffle_epoch)
    iterator_valid_bl = iter(valid_bl.dataset)

    # loop over batches

    print('valid images: ' + str(test_images[:2]))
    for i in tqdm(range(len(test_images) // tc.batch_size)):
        valid_x, valid_y = next(iterator_valid_bl)

        # with tf.device('/cpu:0'):
        with tf.device('/gpu:0'):
            a = time.time()
            valid_output = model_G(valid_x, training=False).numpy()
            target_transformed,_,_ = model_R([valid_y, valid_output])
            b = time.time()

        for j in range(tc.batch_size):
            valid_y_temp = target_transformed.numpy()
            valid_y_temp = valid_y_temp[j]

            valid_x_temp = tf.concat([valid_x[j, :, :, 0:2], valid_x[j, :, :, 3:4]], axis=-1)
            valid_x_temp = (valid_x_temp / tf.reduce_max(valid_x_temp)).numpy()

            valid_image_path = test_images[i * tc.batch_size + j]

            cur_out_img_name = valid_image_path.split('\\')[-1].replace('.mat', '') + '.png'
            cur_out_mat_name = valid_image_path.split('\\')[-1]

            # with case name
            plt.imsave(output_path + '/target_registered/' + cur_out_img_name.replace('.png', '_inp.png'),
                       valid_x_temp[:, :, 0])
            plt.imsave(output_path + '/target_registered/' + cur_out_img_name,
                       valid_y_temp)
            plt.imsave(output_path + '/target_registered/' + cur_out_img_name.replace('.png', '_gt.png'),
                       valid_y.numpy()[j])
            scipy.io.savemat(output_path + '/target_registered/' + cur_out_mat_name, {"target": valid_y_temp})

