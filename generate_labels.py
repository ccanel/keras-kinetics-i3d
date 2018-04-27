'''
Batchification
'''

import numpy as np
import argparse

import tensorflow as tf

def main():
  return 0;

if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-frames', help='Window size.', type=int, required=False, default=79);
    parser.add_argument('--data-dir', help='Data dir', type=str, required=False, default='./data');
    parser.add_argument('--labels', help='File with labels', type=str, required=False, default='./data/labels.dat');

    args = parser.parse_args()
    main(args)
