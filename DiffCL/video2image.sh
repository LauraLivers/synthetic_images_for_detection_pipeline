#!/usr/bin/env bash

# call file with './video2image.sh [input directory path] [output directory path]'
# output folder will be automatically generated if nonexistent

input_dir="$1"
output_dir="$2"

if [[ -z "$input_dir" || -z "$output_dir" ]]; then
    echo "Usage: $0 <input_folder> <output_folder>"
    exit 1
fi

mkdir -p "$output_dir"

for video in "$input_dir"/*.{mkv}; do
    [[ -e "$video" ]] || continue

    filename=$(basename -- "$video")
    name="${filename%.*}"

    echo "Processing: $filename"

    ffmpeg -i "$video" \
        -vsync 0 \
        -vf "select='not(mod(t,10))'" \
        "$output_dir/${name}_%04d.png
done