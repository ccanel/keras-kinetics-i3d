'''
Trains i3d
'''

import numpy as np
import argparse

from i3d_inception import Inception_Inflated3d

from keras import backend as K
from keras.optimizers import Adam

import tensorflow as tf
from tensorflow.python.tools import freeze_graph

import util

FRAME_HEIGHT = 224
FRAME_WIDTH = 224
NUM_RGB_CHANNELS = 3
NUM_FLOW_CHANNELS = 2

NUM_CLASSES = 400

SAMPLE_DATA_PATH = {
    'rgb' : 'data/v_CricketShot_g04_c01_rgb.npy',
    'flow' : 'data/v_CricketShot_g04_c01_flow.npy'
}

LABEL_MAP_PATH = 'data/label_map.txt'
EPOCHS=40

def save_pb(mem_model, prefix):
    sess = K.get_session()
    graph_def = sess.graph.as_graph_def()
    tf.train.write_graph(graph_def,
                         logdir='.',
                         name=prefix+'.pb',
                         as_text=False)
    saver = tf.train.Saver()
    saver.save(sess, prefix+'.ckpt', write_meta_graph=True)

import sys
from sklearn.utils import class_weight

def train(model, video_file, window_size, labels_dir, batch_size, pct_frames):
  optimizer = Adam(0.0001);
  model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
  X, Y = util.get_training_data(video_file, window_size, labels_dir, pct_frames)
  Y_decoded = [y.argmax() for y in Y]
  class_weights = dict(enumerate(class_weight.compute_class_weight(
    'balanced', np.unique(Y_decoded), Y_decoded)))
  print("Class weights: " + str(class_weights))
  model.fit(x=X, y=Y, epochs=EPOCHS, validation_split=0.2, batch_size=batch_size, class_weight=class_weights)

def main(args):
    window_size = args.window;
    # load the kinetics classes
    kinetics_classes = [x.strip() for x in open(LABEL_MAP_PATH, 'r')]


    if args.eval_type in ['rgb', 'joint']:
        if args.no_imagenet_pretrained:
            # build model for RGB data
            # and load pretrained weights (trained on kinetics dataset only) 
            rgb_model = Inception_Inflated3d(
                include_top=False,
                weights='rgb_kinetics_only',
                input_shape=(window_size, FRAME_HEIGHT, FRAME_WIDTH, NUM_RGB_CHANNELS),
                classes=NUM_CLASSES)
        else:
            # build model for RGB data
            # and load pretrained weights (trained on imagenet and kinetics dataset)
            rgb_model = Inception_Inflated3d(
                include_top=False,
                weights='rgb_imagenet_and_kinetics',
                input_shape=(window_size, FRAME_HEIGHT, FRAME_WIDTH, NUM_RGB_CHANNELS),
                classes=NUM_CLASSES)
        train(rgb_model, args.video, window_size, args.labels_dir, args.batch, args.pct_frames)
        save_pb(rgb_model, '/tmp/rgb_model')

        # load RGB sample (just one example)
        rgb_sample = np.load(SAMPLE_DATA_PATH['rgb'])
        rgb_sample = np.asarray([rgb_sample[0][:window_size]])
        
        # make prediction
        rgb_logits = rgb_model.predict(rgb_sample)


    if args.eval_type in ['flow', 'joint']:
        if args.no_imagenet_pretrained:
            # build model for optical flow data
            # and load pretrained weights (trained on kinetics dataset only)
            flow_model = Inception_Inflated3d(
                include_top=False,
                weights='flow_kinetics_only',
                input_shape=(window_size, FRAME_HEIGHT, FRAME_WIDTH, NUM_FLOW_CHANNELS),
                classes=NUM_CLASSES)
        else:
            # build model for optical flow data
            # and load pretrained weights (trained on imagenet and kinetics dataset)
            flow_model = Inception_Inflated3d(
                include_top=False,
                weights='flow_imagenet_and_kinetics',
                input_shape=(window_size, FRAME_HEIGHT, FRAME_WIDTH, NUM_FLOW_CHANNELS),
                classes=NUM_CLASSES)
        train(flow_model, args.video, window_size, args.labels_dir, args.batch, args.pct_frames)
        save_pb(flow_model, '/tmp/flow_model')


        # load flow sample (just one example)
        flow_sample = np.load(SAMPLE_DATA_PATH['flow'])
        flow_sample = np.asarray([flow_sample[0][:window_size]]);
        
        # make prediction
        flow_logits = flow_model.predict(flow_sample)

    # produce final model logits
    if args.eval_type == 'rgb':
        sample_logits = rgb_logits
    elif args.eval_type == 'flow':
        sample_logits = flow_logits
    else: # joint
        sample_logits = rgb_logits + flow_logits

    # produce softmax output from model logit for class probabilities
    sample_logits = sample_logits[0] # we are dealing with just one example
    sample_predictions = np.exp(sample_logits) / np.sum(np.exp(sample_logits))

    sorted_indices = np.argsort(sample_predictions)[::-1]

    print('\nNorm of logits: %f' % np.linalg.norm(sample_logits))
    print('\nTop classes and probabilities')
    sample_predictions = np.squeeze(sample_predictions);
    sample_logits = np.squeeze(sample_predictions);
    for index in np.squeeze(sorted_indices):
      print(sample_predictions[index], sample_logits[index], kinetics_classes[index])

    return


if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--window', help='Window size.', type=int, required=False, default=79);
    parser.add_argument('--eval-type', 
        help='specify model type. 1 stream (rgb or flow) or 2 stream (joint = rgb and flow).', 
        type=str, choices=['rgb', 'flow', 'joint'], default='rgb')
    parser.add_argument('--video', help='Video file.', type=str, required=False, default='./video.mp4')
    parser.add_argument('--labels-dir', help='Labels dir', type=str, required=False, default='./labels')
    parser.add_argument('--batch', help='Batch size.', type=int, required=False, default=1);

    parser.add_argument('--no-imagenet-pretrained',
        help='If set, load model weights trained only on kinetics dataset. Otherwise, load model weights trained on imagenet and kinetics dataset.',
        action='store_true')
    parser.add_argument('--pct-frames', help='Percentage of frames used to train [0, 1.0].', type=float, required=False, default=1.0);

    args = parser.parse_args()
    main(args)
