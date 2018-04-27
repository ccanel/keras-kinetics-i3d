'''
Utils for loading data
'''

import argparse
from collections import Counter
import itertools
import os

import cv2
from keras.utils import np_utils
import numpy as np

RESIZE_DIM = 224

PARTIAL_START_STR = "Partial Start"
PARTIAL_END_STR = "Partial Finish"
FULL_START_STR = "Full Start"
FULL_END_STR = "Full Finish"

STATE_NO_EVENT = "No Event"
STATE_UNKNOWN = "Unknown"
STATE_EVENT = "Event"

LABEL_MAP = { 
  STATE_NO_EVENT: 0,
  STATE_EVENT: 1
}


def video_reader(path):
  print("Reading video from file: {}".format(path))
  count = 0
  cap = cv2.VideoCapture(path)
  if(not cap.isOpened()):
    raise Exception("Violent Error")
  while(cap.isOpened()):
    code, frame = cap.read()
    if(count % 1000 == 0):
      print("Read frame " + str(count))
    count += 1
    if(code):
      yield cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (RESIZE_DIM, RESIZE_DIM))
    else:
      break
  cap.release()


def read_file(filename, data_points):
  print("Reading labels from file: {}".format(filename))
  f = open(filename)
  for line in f:
    split_line = line.rstrip().split(":")
    data_points[int(split_line[1])] = split_line[0]


def get_per_frame_labels(labels_dir):
  data_points = {}
  full_labels = {}
  prev_state = STATE_NO_EVENT
  state = STATE_NO_EVENT
  for f in os.listdir(labels_dir):
    read_file(os.path.join(labels_dir, f), data_points)
    for idx in xrange(sorted(data_points.keys())[-1]):
      if idx in data_points:
        if(data_points[idx] == PARTIAL_START_STR):
          prev_state = STATE_UNKNOWN
          state = STATE_UNKNOWN
        if(data_points[idx] == PARTIAL_END_STR):
          prev_state = STATE_NO_EVENT
          state = STATE_NO_EVENT
        if(data_points[idx] == FULL_START_STR):
          state = STATE_EVENT
        if(data_points[idx] == FULL_END_STR):
          state = prev_state
      full_labels[idx] = state
  for key in sorted(full_labels.keys()):
      val = full_labels[key]
  return data_points, full_labels


def get_data_window(window_size, frames, total_frames):
  for i in range(0, total_frames - window_size, window_size):
    yield np.asarray(list(itertools.islice(frames, window_size)))


def get_label_window(window_size, full_labels):
  labels = []
  total_frames = len(full_labels)
  for i in range(0, total_frames - window_size, window_size):
    cur_window_labels = [full_labels[k] for k in range(i, i + window_size)]
    data = Counter(cur_window_labels)
    labels.append(data.most_common(1)[0][0])
  return labels


def training_generator(data_windows, label_windows):
  for label_window in label_windows:
    data_window = data_windows.next()
    if(label_window != STATE_UNKNOWN):
      yield data_window, np_utils.to_categorical(np.asarray(LABEL_MAP[label_window]), num_classes=2)
 

def batched_training_generator(gen, batch_size):
  ds = []
  ls = []
  for d, l in gen:
    ds.append(d)
    ls.append(l)
    if(len(ds) >= batch_size):
      yield np.asarray(ds), np.asarray(ls)
      ds = []
      ls = []


def create_training_generator(video_file, window_size, labels_dir, batch_size):
  data_points, full_labels = get_per_frame_labels(labels_dir)
  frames = video_reader(video_file)
  frame_windows = get_data_window(window_size, frames, len(full_labels))
  label_windows = get_label_window(window_size, full_labels)
  return batched_training_generator(training_generator(frame_windows, label_windows), batch_size)


def test(args):
  g = create_training_generator(args.video, args.window, args.labels_dir)
  count = 0

if __name__ == '__main__':
  # parse arguments
  parser = argparse.ArgumentParser()
  parser.add_argument('--window', help='Window size.', type=int, required=False, default=16)
  parser.add_argument('--video', help='Video file.', type=str, required=False, default='./video.mp4')
  parser.add_argument('--labels-dir', help='Labels dir', type=str, required=False, default='./labels')
  test(parser.parse_args())
