#!/bin/bash
task=$1
if [[ $1 == "redcar" ]]; then
  VIDEO="/home/ubuntu/mnt/data/scaled_vimba_iii_1_2018-3-21_16.mp4-224-short.mp4"
  LABELS="../labeler/iii_01_redcar/iii_01_16"
elif [[ $1 == "bus" ]]; then
  VIDEO="/home/ubuntu/mnt/data/vimba_iii_1_2018-3-21_7-224-short.mp4"
  LABELS="../labeler/iii_01_buses/iii_01_7"
else
  echo "You must specify either bus or redcar"
  exit
fi
BATCH=128
PCT_FRAMES=0.4
EPOCHS=15
WINDOW=1

echo "[INFO] Window "$WINDOW
python train.py --window $WINDOW \
                --video $VIDEO \
                --eval-type rgb \
                --labels-dir $LABELS \
                --batch $BATCH \
                --epochs $EPOCHS \
                --pct-frames $PCT_FRAMES | tee "window-$WINDOW.dat"

BATCH=16
WINDOW=15

echo "[INFO] Window "$WINDOW
python train.py --window $WINDOW \
                --video $VIDEO \
                --eval-type rgb \
                --labels-dir $LABELS \
                --batch $BATCH \
                --epochs $EPOCHS \
                --pct-frames $PCT_FRAMES | tee "window-$WINDOW.dat"

BATCH=8
WINDOW=30

echo "[INFO] Window "$WINDOW
python train.py --window $WINDOW \
                --video $VIDEO \
                --eval-type rgb \
                --labels-dir $LABELS \
                --batch $BATCH \
                --epochs $EPOCHS \
                --pct-frames $PCT_FRAMES | tee "window-$WINDOW.dat"
