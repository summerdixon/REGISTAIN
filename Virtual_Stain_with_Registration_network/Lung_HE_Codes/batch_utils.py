import random, threading
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
import numpy as np
import scipy.io

from matplotlib import pyplot as plt


class BatchLoader(object):

    def __init__(self, images, image_, label_, config):
        self.images = images
        self.image_ = image_
        self.label_ = label_
        self.config = config
        self.capacity = config.q_limit
        image_shape = self.image_.get_shape().as_list()[1:]
        label_shape = self.label_.get_shape().as_list()[1:]
        self.threads = []
        self.queue = tf.RandomShuffleQueue(shapes=[image_shape, label_shape],
                                           dtypes=[tf.float32, tf.float32],
                                           capacity=self.capacity,
                                           min_after_dequeue=0)
        self.enqueue_op = self.queue.enqueue_many([self.image_, self.label_])

    def get_batch(self):
        return self.queue.dequeue_many(self.config.batch_size)

    def create_thread(self, sess, thread_id, n_threads):
        for image_batch, label_batch in self.batch_generator(self.images[thread_id::n_threads]):
            # try:
            sess.run(self.enqueue_op, feed_dict={self.image_: image_batch, self.label_: label_batch})
            # except:
            #     print('value error')

    def start_threads(self, sess, n_threads):
        for i in range(n_threads):
            thread = threading.Thread(target=self.create_thread, args=(sess, i, n_threads))
            self.threads.append(thread)
            thread.start()


class TrainBatchLoader(BatchLoader):

    def __init__(self, images, image_, label_, config):
        super().__init__(images, image_, label_, config)

    def batch_generator(self, paths):
        size = len(paths)
        s = self.config.image_size
        stride = s
        while True:
            for path in paths:

                # image = np.load(path.replace('target_aligned_r1', 'input_reg'))[20:-20,20:-20,0]

                image_af = scipy.io.loadmat(path.replace('target', 'input'))
                image_af = np.array(image_af['input'], dtype=np.float32)
                image = image_af[20:-20, 20:-20, :]

                # image = np.dstack((np.load(path.replace('amplitude', 'retardance'))[20:-20,20:-20,0],np.load(path.replace('target', 'input'))[20:-20,20:-20,1]))
                # image = scipy.io.loadmat(path.replace('target', 'input'))[20:-20,20:-20,:2]
                # image = np.dstack((image,np.load(path.replace('target_alligned_r2', 'input'))[20:-20,20:-20,1]))
                # image = np.dstack((image,np.load(path.replace('target_alligned_r2', 'input'))[20:-20,20:-20,0]))
                # image = np.array(image, dtype = np.float32)
                # image = np.expand_dims(image,-1)
                # print(path)
                # print(image.shape)

                image_temp = image.copy()
                # # for i in range(4):
                image[:, :] = (image_temp[:, :] - np.mean(image_temp[:, :])) / (np.std(image_temp[:, :]))
                # image[:, :] = image_temp[:, :] / 65535 * 2 - 1
                # label = np.load(path)[20:-20,20:-20,:]
                # label = np.array(label, dtype = np.float32)*255
                label = scipy.io.loadmat(path)
                label = np.array(label['target'], dtype=np.float32) * 255
                label = label[20:-20, 20:-20, :]

                # print(label.shape)
                label = np.nan_to_num(label)
                label_temp = label.copy()
                label[:,:,0] = 0.299*label_temp[:,:,0]+0.587*label_temp[:,:,1]+0.114*label_temp[:,:,2]
                label[:,:,1] = (label_temp[:,:,0] - label[:,:,0])*0.713 + 128
                label[:,:,2] = (label_temp[:,:,2] - label[:,:,0])*0.564 + 128
                size = image.shape[0]
                images, labels = [], []
                x = 0
                while True:
                    y = 0
                    while True:
                        rand_choice_stride = random.randint(0, 15)
                        xx = min(x + rand_choice_stride * s // 16, size - s)
                        yy = min(y + rand_choice_stride * s // 16, size - s)
                        if yy != size - s and xx != size - s:
                            img = image[xx:xx + s, yy:yy + s, :]
                            lab = label[xx:xx + s, yy:yy + s, :]
                            temp_lab = label_temp[xx:xx + s, yy:yy + s, :]
                            if np.mean(temp_lab) < 240:
                                if True:
                                    rand_choice = random.randint(0, 7)
                                    # rand_choice=5

                                    if rand_choice == 0:
                                        img = np.fliplr(img)
                                        lab = np.fliplr(lab)
                                    elif rand_choice == 1:
                                        img = np.flipud(img)
                                        lab = np.flipud(lab)
                                    elif rand_choice == 2:
                                        img = np.rot90(img, k=1)
                                        lab = np.rot90(lab, k=1)
                                    elif rand_choice == 3:
                                        img = np.rot90(img, k=2)
                                        lab = np.rot90(lab, k=2)
                                    elif rand_choice == 4:
                                        img = np.rot90(img, k=3)
                                        lab = np.rot90(lab, k=3)
                                    elif rand_choice == 5:
                                        img = np.rot90(img, k=1)
                                        img = np.fliplr(img)
                                        lab = np.rot90(lab, k=1)
                                        lab = np.fliplr(lab)
                                    elif rand_choice == 6:
                                        img = np.rot90(img, k=1)
                                        img = np.flipud(img)
                                        lab = np.rot90(lab, k=1)
                                        lab = np.flipud(lab)
                                    # elif rand_choice==5:
                                    #        img = np.rot90(img, k=0)
                                    #        lab = np.rot90(lab, k=0)
                                    images.append(img)
                                    labels.append(lab)
                                    # print(img.shape)
                                    # print(lab.shape)

                        if yy == size - s:
                            break
                        y += stride
                    if xx == size - s:
                        break
                    x += stride
                try:
                    images[0].shape
                    yield np.array(images), np.array(labels)
                except IndexError:
                    print('training image not passed through')


class ValidBatchLoader(BatchLoader):

    def __init__(self, images, image_, label_, config):
        super().__init__(images, image_, label_, config)

    def batch_generator(self, paths):
        size = len(paths)
        s = self.config.image_size
        stride = s
        while True:
            images, labels = [], []
            for path in paths:
                # image = np.dstack((np.load(path.replace('target', 'input'))[20:-20,20:-20,0],np.load(path.replace('target', 'input'))[20:-20,20:-20,1]))

                image_af = scipy.io.loadmat(path.replace('target', 'input'))
                image_af = np.array(image_af['input'], dtype=np.float32)
                image = image_af[20:-20, 20:-20, :]

                # image = np.load(path.replace('target', 'input'))[20:-20,20:-20,:2]
                # image = np.dstack((image,np.load(path.replace('target_alligned_r2', 'input'))[20:-20,20:-20,1]))
                # image = np.dstack((image,np.load(path.replace('target_alligned_r2', 'input'))[20:-20,20:-20,0]))
                # image = np.load(path.replace('target', 'input'))[20:-20,20:-20,4]
                # print(image.shape)
                image = np.array(image, dtype=np.float32)
                # image = np.expand_dims(image,-1)
                image_temp = image.copy()
                # for i in range(4):
                image[:, :] = (image_temp[:, :] - np.mean(image_temp[:, :])) / (np.std(image_temp[:, :]))
                # image[:, :] = image_temp[:, :] / 65535 * 2 - 1
                # label = np.load(path)[20:-20,20:-20,:]#[20:-20,20:-20,:]
                # label = np.array(label, dtype = np.float32)*255

                label = scipy.io.loadmat(path)
                label = np.array(label['target'], dtype=np.float32) * 255
                label = label[20:-20, 20:-20, :]

                label = np.nan_to_num(label)
                label_temp = label.copy()
                label[:,:,0] = 0.299*label_temp[:,:,0]+0.587*label_temp[:,:,1]+0.114*label_temp[:,:,2]
                label[:,:,1] = (label_temp[:,:,0] - label[:,:,0])*0.713 + 128
                label[:,:,2] = (label_temp[:,:,2] - label[:,:,0])*0.564 + 128

                size = image.shape[0] - 1
                x = 0
                while True:
                    y = 0
                    while True:
                        xx = min(x, size - s)
                        yy = min(y, size - s)
                        images.append(image[xx:xx + s, yy:yy + s, :])
                        labels.append(label[xx:xx + s, yy:yy + s, :])

                        if yy == size - s:
                            break
                        y += stride
                    if xx == size - s:
                        break
                    x += stride
            # print(len(images))
            # exit()
            try:
                images[0].shape
                yield np.array(images), np.array(labels)
            except IndexError:
                print(' valid image not passed through')

