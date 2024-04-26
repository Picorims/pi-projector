#!/bin/bash

# create dir if it does not exist
if [ ! -d "./tmp" ]; then
  echo "./tmp does not exist. creating it..."
  mkdir tmp
fi


# convert
echo converting... =======================
ffmpeg -y -i "$1" tmp/sample.mp4
# downscale 100px
echo downscaling... ==========================
ffmpeg -y -i tmp/sample.mp4 -filter:v scale=100:-1 -c:a copy tmp/downscale.mp4
# framerate 10fps
echo changing framerate... =========================
ffmpeg -y -i tmp/downscale.mp4 -filter:v fps=10 tmp/downscale_10fps.mp4
# to 100x100 with black pixels if needed
echo add black pixels if needed... =====================
ffmpeg -y -i tmp/downscale_10fps.mp4 -vf "scale=100:100:force_original_aspect_ratio=decrease,pad=100:100:(ow-iw)/2:(oh-ih)/2,setsar=1" tmp/projector_input.mp4
echo ready! ===========================

# reading
./read_pixels_parall.out ./tmp/projector_input.mp4 --full-cache