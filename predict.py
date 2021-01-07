# USAGE
# python predict.py --image ../data/infer_test_images/full_visibility.png --artifact ../artifacts/idvggnet --labels ../artifacts/idvggnet/idvggnet_lb.pickle 

# import the necessary packages
# from tensorflow.keras.models import load_model
from tensorflow.keras.models import load_model
import argparse
import pickle
import cv2
import logging
import numpy as np
from nn.customvggnet import feature_engineer_to_train
import sys

logging.basicConfig(level=logging.INFO, stream=sys.stdout, format='%(levelname)s %(filename)s:%(lineno)4d: %(message)s')

# construct the argument parser and parse the arguments
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--image', required=True, help='path to input image for inference test')
    ap.add_argument('--artifact', required=True, help='path to trained model')
    ap.add_argument('--labels', required=True, help='path to labels file')
    ap.add_argument('--dim', type=int, default=64, help='expected dimxdim input model')
    ap.add_argument('--median', type=int, default=17, help='median filter kernel size, should be odd')
    args = ap.parse_args()
    return args


def invoke(args):
    # Load trained model and label file
    logging.info('Loading model and label file...')
    model = load_model(args.artifact)
    lb = pickle.loads(open(args.labels, 'rb').read())
    # Load and pre process image
    #image = cv2.imread(args.image, cv2.IMREAD_UNCHANGED)
    image = cv2.imread(args.image) # will load image with defaults; if rgba or gray then transforms to 3-channel
    if type(image) is not np.ndarray:
        logging.info('Image not found')
        return
    img_infer = feature_engineer_to_train(image, args.median, args.dim)
    if type(img_infer) is not np.ndarray:
        logging.info('Image is not 3-channel')
        return
    # infer on image
    img_infer = img_infer.reshape((1, img_infer.shape[0], img_infer.shape[1], img_infer.shape[2]))
    preds = model.predict(img_infer, args.median, args.dim)
    # get class prediction high highest score
    i = preds.argmax(axis=1)[0]
    label = lb.classes_[i]
    # print predictions
    print(f'class:{label}, prob: {preds[0][i] * 100}')
    # show result image superimposing the prediction result
    text = "{}: {:.2f}%".format(label, preds[0][i] * 100)
    cv2.putText(image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 2)
    # show the output image
    cv2.imshow('Visibility result - press a key to exit', image)
    cv2.waitKey(0)


if __name__ == "__main__":
    args = parse_args()
    invoke(args)

