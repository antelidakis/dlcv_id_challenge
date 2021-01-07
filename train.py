# USAGE
# For upsampling method
# python train.py --annot path/to/annot.csv --output path/to/artifacts/idvggnet_demo --upsampling 1
# For standard pipeline method that assumes balanced dataset
# python train.py --annot path/to/annot.csv --output path/to/artifacts/idvggnet_demo
#
# e.g.:
# python train.py --annot /Users/antonios.ntelidakis/Documents/personal/revolut/Computer vision HT/data/gicsd_labels.csv --output /Users/antonios.ntelidakis/Documents/dev/antonios_ntelidakis_ht/artifacts/idvggnet_upsampling_test/
#
# Please advise : ../notebooks/c_i_train_cnn_grayscale_jit_data_gen.ipynb for the code module details and work through of the approach

from matplotlib import pyplot as plt
# import the necessary packages
import sys
# sys.path.append('/Users/antonios.ntelidakis/Documents/dev/keras_grayscale_idpose_custom')
from nn.customvggnet import ACustomVGGNet, feature_engineer_to_train
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import pickle
import cv2
import os
import csv
import logging
import random
from itertools import cycle, islice


logging.basicConfig(level=logging.INFO, stream=sys.stdout, format='%(levelname)s %(filename)s:%(lineno)4d: %(message)s')

# construct the argument parser and parse the arguments
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--annot', required=True, help='path to annotation csv file')
    ap.add_argument('--output', required=True, help='artifact output directory')
    # Image preprocssing params
    ap.add_argument('--dim', type=int, default=64, help='expected dimxdim input model')
    ap.add_argument('--median', type=int, default=17, help='median filter kernel size, should be odd')
    # Training params
    ap.add_argument('--upsampling', type=int, default=-1, help='Set to 1 to train with upsampling')
    ap.add_argument('--epochs', type=int, default=75, help='number of epochs in training')
    ap.add_argument('--test_size', type=float, default=0.25, help='number of epochs in training')
    ap.add_argument('--depth', type=int, default=1, help='Set VGGs expected input image depth')
    ap.add_argument('--bs', type=int, default=32, help='Set VGGs training batchsize')
    ap.add_argument('--lr', type=int, default=0.01, help='Set VGGs learning rate')
    args = ap.parse_args()
    return args


def shuffle_and_partition_train_test(raw, test_percent):
    howManyNumbers = int(round(test_percent*len(raw)))
    shuffled = raw[:]
    random.shuffle(shuffled)
    return shuffled[howManyNumbers:], shuffled[:howManyNumbers]


def upsambled_dataset(csv_dir, test_size, median, dim):
    # This code is targeting the current ML case study. The dataset has great imbalance. We use an upsampling approach.
    # There was not time to make this bit of the code re-usable fully reusable. 
    # In essens what is left is an automatic way of finding the unique classes, and use a dictionaries where for the key,pair elemet
    # would be 'class_label', 'list_of_paths'
    #   * Load csv, get annotations, split train/test by class type in different lists
    #   * Upsamble each train/test list to match the class with the highest number of examples
    # Please advise the ../notebooks/c_i_train_cnn_grayscale_jit_data_gen.ipynb for exploration of the code and the approach
    
    logging.info(f'Preparing Dataset...')
    base_data_dir = os.path.dirname(csv_dir)
    path_imgs_full_vis = []
    path_imgs_partial_vis = []
    path_imgs_no_vis = []
    test_size=0.25
    #  Load csv, get annotations, append distinct class lists - To-Do should use a dictionary <cls,list_paths>
    with open(csv_dir, newline='') as csvfile:
        labelreader = csv.reader(csvfile, delimiter=',', quotechar='"')
        next(labelreader)
        for row in labelreader:
            src = base_data_dir+'/images/'+row[0]
            cur_label = row[1].strip()
            if cur_label == 'NO_VISIBILITY':
                path_imgs_no_vis.append(src)
            elif cur_label == 'PARTIAL_VISIBILITY':
                path_imgs_partial_vis.append(src)
            elif cur_label == 'FULL_VISIBILITY':
                path_imgs_full_vis.append(src)
    # Split to train valid test each image path class list independently however using the same split ratio
    random.seed(42)
    path_full_vis_train, path_full_vis_test = shuffle_and_partition_train_test(path_imgs_full_vis, test_size)
    path_partial_vis_train, path_partial_vis_test = shuffle_and_partition_train_test(path_imgs_partial_vis, test_size)
    path_no_vis_train, path_no_vis_test = shuffle_and_partition_train_test(path_imgs_no_vis, test_size)
    # Print dataset details
    logging.info(f'Dataset train/test examples per class')
    print(f'FULL_VISIBILITY. n_train: {len(path_full_vis_train)}, n_test: {len(path_full_vis_test)}, tota: {len(path_imgs_full_vis)}')
    print(f'PARTIAL_VISIBILITY. n_train: {len(path_partial_vis_train)}, n_test: {len(path_partial_vis_test)}, tota: {len(path_imgs_partial_vis)}')
    print(f'NO_VISIBILITY. n_train: {len(path_no_vis_train)}, n_test: {len(path_no_vis_test)}, tota: {len(path_imgs_no_vis)}')
    # Find class with max number of train, test examples, and use that number to repeat/ upsamble train, test data
    max_train = max(len(path_full_vis_train), len(path_partial_vis_train), len(path_no_vis_train))
    max_test = max(len(path_full_vis_test), len(path_partial_vis_test), len(path_no_vis_test))
    print(f'Max Train: {max_train}, Max Test: {max_test}')
    logging.info(f'Upsampling the weak classes to match the dominant class examples and create final train/test data')
    # Upsample path Data
    path_full_vis_train = list(islice(cycle(path_full_vis_train), max_train))
    path_partial_vis_train = list(islice(cycle(path_partial_vis_train), max_train))
    path_no_vis_train = list(islice(cycle(path_no_vis_train), max_train))
    path_full_vis_test = list(islice(cycle(path_full_vis_test), max_test))
    path_partial_vis_test = list(islice(cycle(path_partial_vis_test), max_test))
    path_no_vis_test = list(islice(cycle(path_no_vis_test), max_test))
    # Load actual train and test data upsampled in a format ready training
    # Load Train
    tota_path_train = path_full_vis_train + path_partial_vis_train + path_no_vis_train
    trainX = []
    trainY = ['FULL_VISIBILITY']*max_train + ['PARTIAL_VISIBILITY']*max_train + ['NO_VISIBILITY']*max_train
    for img_path in tota_path_train:
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        img_trainready = feature_engineer_to_train(img)
        trainX.append(img_trainready)    
    trainX = np.array(trainX, dtype="float")
    trainY = np.array(trainY)
    logging.info(f'trainX shape: {trainX.shape}')
    logging.info(f'trainY shape: {trainY.shape}')
    # Load test
    tota_path_test = path_full_vis_test + path_partial_vis_test + path_no_vis_test
    testX = []
    testY = ['FULL_VISIBILITY']*max_test + ['PARTIAL_VISIBILITY']*max_test + ['NO_VISIBILITY']*max_test
    for img_path in tota_path_test:
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        img_trainready = feature_engineer_to_train(img)
        testX.append(img_trainready)  
    testX = np.array(testX, dtype="float")
    testY = np.array(testY)
    logging.info(f'testX shape: {testX.shape}')
    logging.info(f'testY shape: {testY.shape}')
    # Finally do an additional shuffling of the train/test data/label pairs
    indices = np.arange(trainX.shape[0])
    np.random.shuffle(indices)
    trainX = trainX[indices]
    trainY = trainY[indices]
    
    indices = np.arange(testX.shape[0])
    np.random.shuffle(indices)
    testX = testX[indices]
    testY = testY[indices]
    
    return  trainX, testX, trainY, testY 


def original_dataset(csv_dir, test_size, median, dim):
    logging.info(f'Preparing Dataset...')
    base_data_dir = os.path.dirname(csv_dir)
    data = []
    labels = []
    with open(csv_dir, newline='') as csvfile:
        labelreader = csv.reader(csvfile, delimiter=',', quotechar='"')
        next(labelreader)
        for row in labelreader:
            img_path = base_data_dir+'/images/'+row[0]
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            img_trainready = feature_engineer_to_train(img, median, dim)
            data.append(img_trainready)
            cur_label = row[1].strip()
            labels.append(cur_label)
    data = np.array(data, dtype="float")
    labels = np.array(labels)
    logging.info(f'data shape: {data.shape}')
    logging.info(f'labels shape: {labels.shape}')
    (trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=test_size, random_state=42)
    return trainX, testX, trainY, testY


def train_model(trainX, testX, trainY, testY, args):
    # convert the labels from integers to vectors 
    lb = LabelBinarizer()
    trainY = lb.fit_transform(trainY)
    testY = lb.transform(testY)
    logging.info(f'Model classes: {lb.classes_}')
    # Set data augmentation parameters
    # Should probably remove zoom, width and height shift as this may crop edge cases of fully visible images.
    aug = ImageDataGenerator(rotation_range=30, shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode="nearest")
    # initialize our VGG-like Convolutional Neural Network
    model = ACustomVGGNet.build(width=args.dim, height=args.dim, depth=args.depth, classes=len(lb.classes_))
    logging.info(f'Model architecture')
    model.summary()
    # initialize our initial learning rate, num of epochs (how many times we see all train examples) to train for, and batch size 
    INIT_LR = args.lr #0.01
    EPOCHS = args.epochs #75 # 2 # 75
    BS = args.bs #32
    # initialize the model and optimizer 
    opt = SGD(lr=INIT_LR, decay=INIT_LR / EPOCHS)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
    # train the network
    logging.info(f'training network...')
    logging.info(f'INIT_LR:{INIT_LR}, EPOCHS: {EPOCHS}, BS: {BS}')
    H = model.fit_generator(aug.flow(trainX, trainY, batch_size=BS),
        validation_data=(testX, testY), steps_per_epoch=len(trainX) // BS,
        epochs=EPOCHS)
    return model, lb, H, trainY, testY


def invoke(args):
    # 1. Load and prepare training data
    trainX, testX, trainY, testY = [], [], [], []
    if args.upsampling != -1:
        msg_train_method = 'Upsampling imbalanced dataset'
        logging.info(f'Dataset preparation: {msg_train_method!r}')
        trainX, testX, trainY, testY = upsambled_dataset(args.annot, args.test_size, args.median, args.dim)
    else: 
        msg_train_method = 'Standard balanced dataset'
        logging.info(f'Dataset preparation: {msg_train_method!r}')
        trainX, testX, trainY, testY = original_dataset(args.annot, args.test_size, args.median, args.dim)

    # 2. Train model
    model, lb, H, trainY, testY = train_model(trainX, testX, trainY, testY, args)

    # 3. evaluate the network
    logging.info("evaluating network...")
    predictions = model.predict(testX, batch_size=32)
    #print in console scores
    print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=lb.classes_))
    # create a plot (training loss and accuracy) and save it as an image to output artifact folder
    plot_save_dir = args.output + '/training_stats.png'
    os.makedirs(os.path.dirname(plot_save_dir), exist_ok=True)
    N = np.arange(0,args.epochs)
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(N, H.history["loss"], label="train_loss")
    plt.plot(N, H.history["val_loss"], label="val_loss")
    plt.plot(N, H.history["accuracy"], label="train_acc")
    plt.plot(N, H.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy (IDVGGNet)")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    plt.savefig(plot_save_dir)

    # 4. Save model artifact
    # save the model and label binarizer to disk
    logging.info(f'Saving artifact to:{args.output}')
    model.save(args.output)
    pickle_labels_save_dir = args.output + '/idvggnet_lb.pickle'
    f = open(pickle_labels_save_dir, "wb")
    f.write(pickle.dumps(lb))
    f.close()
    

if __name__ == "__main__":
    args = parse_args()
    invoke(args)