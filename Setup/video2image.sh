#!/usr/bin/env bash

# call file with './video2image.sh [input directory path] [output directory path]'
# output folder will be automatically generated if nonexistent
set -x
input_dir="$1"
output_dir="$2"

if [[ -z "$input_dir" || -z "$output_dir" ]]; then
    echo "Usage: $0 <input_folder> <output_folder>"
    exit 1
fi

mkdir -p "$output_dir"

for video in "$input_dir"/*.mkv; do
    [[ -e "$video" ]] || continue

    filename=$(basename -- "$video")
    name="${filename%.*}"

    echo "Processing: $filename"
    fps=$(ffprobe -v error -select_streams v:0 -show_entries stream=r_frame_rate \
    -of csv=p=0 "$video" | bc -l)
    interval=$(echo "$fps * 10" | bc | cut -d. -f1)

    ffmpeg -i "$video" \
        -vf "select='not(mod(n\,${interval}))', setpts=N/FRAME_RATE/TB" \
        -fps_mode vfr \
        "$output_dir/${name}_%04d.png"
done