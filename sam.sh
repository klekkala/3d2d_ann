#!/bin/bash

# Define the directory containing the files
data_dir="./assets/input/"

# Loop through each file in the directory
for file in "$data_dir"/*; do
    # Check if the file is a regular file
    if [ -f "$file" ]; then
        # Extract filename without extension
        filename=$(basename -- "$file")
        filename_no_ext="${filename%.*}"

        # Run the command for each file
        python grounded_sam_demo.py \
            --config GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
            --grounded_checkpoint groundingdino_swint_ogc.pth \
            --sam_checkpoint sam_vit_h_4b8939.pth \
            --input_image "$file" \
            --output_dir "outputs/res/$filename_no_ext" \
            --box_threshold 0.3 \
            --text_threshold 0.25 \
            --text_prompt "building,tree,window,street lamp,bush,door,pavement,person,fire hydrant,leaf,chair,ground,bicycle rack,umbrella" \
            --device "cuda"
    fi
done

