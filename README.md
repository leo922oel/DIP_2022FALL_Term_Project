# Term Project - Color Enhancement
> Digital Image Processing Fall 2022

## Environment
> python `3.8`

    # If you have conda, you can build a conda environment called "dip"
    conda activate dip
    pip install -r requirements.txt

    # Otherwise
    pip install -r requirements.in

## Usage
```
python color_enhance.py \
    --img {path to the original image.}\
    --use_HC {bool; Whether use Histogram Correction or not.} \
    --mean {Mean for Histogram Correction.}\
    --std {Standard deviation for Histogram Correction.}\
    --sim_dim {bool; Whether simulate image with dim backlight or not.} \
    --use_CE {bool; Whether use Color Enhancement or not.} \
    --verbosity {bool; Whether show the result.} \
    --save {bool; Whether save the result.} \
    --output_name {Name of output image} \
```
