ffmpeg -ss 00:00:00 -i vimba_iii_1_2018-3-21_7_scaled.mp4  -vf scale=224:224 -codec:v h264_nvenc -t 01:51:6.666666666666666666 vimba_iii_1_2018-3-21_7-224-short.mp4
