#!/bin/zsh
source activate mousepose
jupyter nbconvert --to script IMP_ir_005_Process_color_images.ipynb
jupyter nbconvert --to script IMP_ir_006_Process_depth_images.ipynb
jupyter nbconvert --to script IMP_ir_007_Auto_Tracking_engine.ipynb
python IMP_ir_005_Process_color_images.py
python IMP_ir_006_Process_depth_images.py
python IMP_ir_007_Auto_Tracking_engine.py
