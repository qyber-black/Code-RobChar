#!/usr/bin/env bash
python generate_example_fig1.py
python generate_fig3.py 
python generate_fig4_kendallrankanalysis.py 
python generate_arim_all_fig5.py 
python gen_fig_8_arim_fcall_scaling.py

cd gray_scale_adjusted_paperfigs || exit 1
bash convert_to_gray.sh

