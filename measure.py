"""
Evaluates the performance of evaluating the I3D model.
"""

import argparse
import time

import numpy as np

from i3d_inception import Inception_Inflated3d


FRAME_HEIGHT = 224
FRAME_WIDTH = 224
NUM_RGB_CHANNELS = 3
NUM_CLASSES = 2
SAMPLE_DATA_PATH = "data/v_CricketShot_g04_c01_rgb.npy"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--window", help="Window size", nargs="+", type=int, required=True)
    parser.add_argument(
        "--samples", help="Num samples", type=int, required=False, default=5)
    parser.add_argument(
        "--out-file", help="Output file", type=str, required=True)
    return parser.parse_args()


def record_result(out_file, window_size, dur):
    msg = "{},{}".format(window_size, dur)
    print("Result: {}".format(msg))
    out_file.write("{}\n".format(msg))


def main():
    args = parse_args()

    samples = args.samples
    print("Number of samples: {}".format(samples))
    with open(args.out_file, "w") as out_file:
        out_file.write("# Window size, time (ms)")

        for window_size in args.window:
            print("Timing window size: {}".format(window_size))

            # build model for RGB data and load pretrained weights (trained on
            # imagenet and kinetics dataset)
            model = Inception_Inflated3d(
                include_top=False,
                weights="rgb_imagenet_and_kinetics",
                input_shape=(window_size, FRAME_HEIGHT, FRAME_WIDTH,
                             NUM_RGB_CHANNELS),
                classes=NUM_CLASSES)

            # load RGB sample (just one example)
            raw_samples = np.load(SAMPLE_DATA_PATH)
            one_sample = raw_samples[0][0]
            all_samples = np.asarray([[one_sample] * window_size])

            start_time = time.time()
            for _ in xrange(samples):
                # make prediction
                model.predict(all_samples)
            dur = (time.time() - start_time) / float(samples)
            record_result(out_file, window_size, dur)
    return


if __name__ == "__main__":
    main()
