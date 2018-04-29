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
BATCH=1
PCT_FRAMES=0.4
EPOCHS=15
WINDOW=180

echo "[INFO] Window "$WINDOW
python train.py --window $WINDOW \
                --video $VIDEO \
                --eval-type rgb \
                --labels-dir $LABELS \
                --batch $BATCH \
                --epochs $EPOCHS \
                --pct-frames $PCT_FRAMES | tee "window-$WINDOW.dat"

BATCH=1
WINDOW=240

echo "[INFO] Window "$WINDOW
python train.py --window $WINDOW \
                --video $VIDEO \
                --eval-type rgb \
                --labels-dir $LABELS \
                --batch $BATCH \
                --epochs $EPOCHS \
                --pct-frames $PCT_FRAMES | tee "window-$WINDOW.dat"

BATCH=1
WINDOW=300

echo "[INFO] Window "$WINDOW
python train.py --window $WINDOW \
                --video $VIDEO \
                --eval-type rgb \
                --labels-dir $LABELS \
                --batch $BATCH \
                --epochs $EPOCHS \
                --pct-frames $PCT_FRAMES | tee "window-$WINDOW.dat"
