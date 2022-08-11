import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import confusion_matrix

import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import pickle

def min_img_size(datadir):
    w, h = (2000, 2000)
    for root, dirs, files in os.walk(datadir):
        for name in files:
            if '.png' in name:
                filename = os.path.join(root, name)
                img = Image.open(filename)
                new_w, new_h = img.size
                if new_w < w:
                    w = new_w
                if new_h < h:
                    h = new_h
    img_size = (h, w)
    return img_size

if __name__ == '__main__':

    # change model (and file_name)
    from 03_model_generation import *
    model_name = os.path.basename(__file__)[:-3]  # [:-3] to remove the '.py' (the last 3 characters)

    # choose the right paths (run either on my machine or on UBELIX):

    #saving_path = "/Users/saskia/unibe19/master_thesis/TKI_project/UBELIX/models/"
    #traindir = "/Users/saskia/unibe19/master_thesis/TKI_project/data/dataset3class_NS_IL1bHigh_TNF/train/"
    #testdir = "/Users/saskia/unibe19/master_thesis/TKI_project/data/dataset3class_NS_IL1bHigh_TNF/testset192/"

    saving_path = "/storage/homefs/sp11d047/TKI_project/models/"
    traindir ="/storage/homefs/sp11d047/TKI_project/data/dataset3class_NS_IL1bHigh_TNF/train/"
    testdir = "/storage/homefs/sp11d047/TKI_project/data/dataset3class_NS_IL1bHigh_TNF/testset192/"

    # some parameters:
    crop_sz = 192
    batch_size = 32
    n_class = 3

    # load data

    img_size = min_img_size(traindir)

    trainset = tf.keras.utils.image_dataset_from_directory(traindir, color_mode='grayscale',
                                                           batch_size=1, image_size=img_size)
    testset = tf.keras.utils.image_dataset_from_directory(testdir, color_mode='grayscale',
                                                          batch_size=batch_size, image_size=(crop_sz, crop_sz))

    normalization_layer = tf.keras.layers.Rescaling(1. / 255)
    trainset = trainset.map(lambda im, lbl: (normalization_layer(im), lbl))
    testset = testset.map(lambda im, lbl: (normalization_layer(im), lbl))

    trainset = trainset.repeat(count=40)

    # crop and data augmentation of the trainset

    rand_transform = keras.models.Sequential([
        keras.layers.RandomCrop(crop_sz * 2, crop_sz * 2),
        keras.layers.RandomRotation(1, fill_mode='reflect'),
        keras.layers.CenterCrop(crop_sz, crop_sz),
        keras.layers.RandomFlip(mode='horizontal'),
        keras.layers.RandomContrast(factor=0.1)
    ])

    def augment_dataset(img, lbl):
        img_aug = rand_transform.call(img, training = True)
        return img_aug[0], lbl[0]

    trainset = trainset.map(augment_dataset)

    trainset = trainset.batch(batch_size)
    trainset = trainset.prefetch(128)

    img_shape = (crop_sz, crop_sz, 1)
    model = gen_model(n_class, img_shape)

    # save model summary to file:
    # open the file
    with open(os.path.join(saving_path, model_name + '_summary.txt'), 'w') as fh:
        # pass the file handle in as a lambda function to make it callable
        model.summary(print_fn=lambda x: fh.write(x + '\n'))

    model_path = os.path.join(saving_path, model_name + '_{epoch}.h5')
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(model_path, save_best_only=False)

    # change epochs number (350)
    hist = model.fit(trainset, epochs=350, validation_data=testset,
                     callbacks=[checkpoint_cb])

    # pickle hist.history
    p_file_path = os.path.join(saving_path, model_name + '_hist.pckl')
    pickle.dump(hist.history, open(p_file_path, 'wb'))

    transparency=0.3
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].plot(hist.epoch, hist.history['loss'], alpha=transparency)
    axs[0].plot(hist.epoch, hist.history['val_loss'], alpha=transparency)
    axs[0].legend(('training loss', 'validation loss'), loc='lower right')
    axs[1].plot(hist.epoch, hist.history['accuracy'], alpha=transparency)
    axs[1].plot(hist.epoch, hist.history['val_accuracy'], alpha=transparency)
    axs[1].legend(('training accuracy', 'validation accuracy'), loc='lower right')

    # change path for saving figure
    plt.savefig(os.path.join(saving_path, model_name + '_training.png'))

    # restore best model
    best_idx = np.argmax(hist.history['val_accuracy']) + 1
    model = tf.keras.models.load_model(os.path.join(saving_path, model_name + f'_{best_idx}.h5'))

    # save best-epoch model
    tf.keras.models.save_model(model, os.path.join(saving_path, model_name + f'_bestEpoch.h5'))

    # open summary file with access mode 'a'
    with open(os.path.join(saving_path, model_name + '_summary.txt'), 'a') as file_object:
        # append best epoch at the end of file
        file_object.write(f'id of best epoch: {best_idx}')

    # retrieve labels
    Xs = []
    Ys = []
    for x, y in testset:
        Xs.append(x.numpy())
        Ys.append(y.numpy())

    Xs = np.concatenate(Xs, axis=0)
    Ys = np.concatenate(Ys, axis=0)

    # prediction
    probs = model.predict(Xs)
    pred = np.argmax(probs, axis=1)

    # confusion matrix
    conf_mx = confusion_matrix(Ys, pred, normalize='true')  # normalize='all'

    fig, ax = plt.subplots(1, 1)

    img = ax.matshow(conf_mx, vmin=0., vmax=1.)

    ax.set_xticks(range(0, 3))
    ax.set_yticks(range(0, 3))

    ax.set_title('confusion matrix')
    ax.set_xlabel('predicted label')
    ax.set_ylabel('true label')

    class_names = ['NS', 'IL-1b', 'TNF']

    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)

    fig.colorbar(img)

    # change path for saving figure
    plt.savefig(os.path.join(saving_path, model_name + '_CM.png'))
