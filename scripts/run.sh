VIDEO="/home/ubuntu/mnt/data/vimba_iii_1_2018-3-21_7-224-short.mp4"
LABELS="../labeler/iii_01_buses/iii_01_7"
BATCH=16
PCT_FRAMES=0.4
EPOCHS=3
WINDOW=2

echo "[INFO] Window "$WINDOW
python train.py --window $WINDOW \
                --video $VIDEO \
                --eval-type rgb \
                --labels-dir $LABELS \
                --batch $BATCH \
                --epochs $EPOCHS \
                --pct-frames $PCT_FRAMES
