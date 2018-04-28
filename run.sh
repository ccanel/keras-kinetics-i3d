VIDEO="/home/ubuntu/mnt/data/vimba_iii_1_2018-3-21_7-224-short.mp4"
WINDOW=9
LABELS="../labeler/iii_01_buses/iii_01_7"
BATCH=16
PCT_FRAMES=0.2
EPOCHS=2
python train.py --window $WINDOW \
                --video $VIDEO \
                --eval-type rgb \
                --labels-dir $LABELS \
                --batch $BATCH \
                --epochs $EPOCHS \
                --pct-frames $PCT_FRAMES
