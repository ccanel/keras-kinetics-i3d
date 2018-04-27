'''
Batchification
'''

import numpy as np
import argparse
import os
import itertools

import tensorflow as tf
import cv2

from collections import Counter

def video_reader(path):
  count = 0;
  cap = cv2.VideoCapture(path)
  if(cap.isOpened() == False):
    raise Exception("Violent Error")
  while(cap.isOpened()):
    code, frame = cap.read()
    print("Read frame " + str(count))
    count += 1;
    if(code == True):
      yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    else:
      break
  cap.release()

def read_file(filename, data_points):
  print(filename)
  f = open(filename)
  for line in f:
    split_line = line.rstrip().split(":")
    data_points[int(split_line[1])] = split_line[0]

partial_start_str = "Partial Start"
partial_end_str = "Partial Finish"
full_start_str = "Full Start"
full_end_str = "Full Finish"

state_no_event = "No Event"
state_unknown = "Unknown"
state_event = "Event"

def get_per_frame_labels(labels_dir):
  data_points = dict()
  full_labels = dict()
  prev_state = state_no_event
  state = state_no_event
  for f in os.listdir(labels_dir):
    read_file(os.path.join(labels_dir, f), data_points)
    for idx in range(0, sorted(data_points.keys())[-1]):
      if idx in data_points:
        if(data_points[idx] == partial_start_str):
          prev_state = state_unknown
          state = state_unknown
        if(data_points[idx] == partial_end_str):
          prev_state = state_no_event
          state = state_no_event
        if(data_points[idx] == full_start_str):
          state = state_event
        if(data_points[idx] == full_end_str):
          state = prev_state
      full_labels[idx] = state
  for key in sorted(full_labels.keys()):
      val = full_labels[key]
  return data_points, full_labels


def read_video_file(path):
  count = 0
  cap = cv2.VideoCapture(path)
  frames = []
  if(cap.isOpened() == False):
    raise Exception("Violent Error")
  while(cap.isOpened()):
    code, frame = cap.read()
    print("Read frame " + str(count));
    count += 1;
    if(code == True):
      frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
      frames.append(frame)
    else:
      break
  cap.release()
  return frames

def get_data(window_size, frames, total_frames):
  for i in range(0, total_frames - window_size, window_size):
    yield itertools.islice(frames, window_size)

def get_labels(window_size, full_labels):
  labels = []
  total_frames = len(full_labels)
  for i in range(0, total_frames - window_size, window_size):
    cur_window_labels = [full_labels[k] for k in range(i, i + window_size)]
    data = Counter(cur_window_labels)
    labels.append(data.most_common(1)[0])
  return labels

def training_generator(data_gen, labels):
  for label in labels:
    yield data_gen.next(), np.asarray(labels)
    
def create_training_generator(video_file, window_size, labels_dir):
  data_points, full_labels = get_per_frame_labels(labels_dir)
  frames = video_reader(video_file)
  labels = get_labels(window_size, full_labels)
  return training_generator(frames, labels);

def main(args):
  data_points, full_labels = get_per_frame_labels()
  frames = video_reader(args.video)
  labels = get_labels(args.window, full_labels)
  g = get_data(args.window, frames, len(full_labels))
  print(list(g.next()))
  print(len(labels))
  return 0

if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--window', help='Window size.', type=int, required=False, default=16)
    parser.add_argument('--video', help='Video file.', type=str, required=False, default='./video.mp4')
    parser.add_argument('--labels-dir', help='Labels dir', type=str, required=False, default='./labels')

    args = parser.parse_args()
    main(args)
