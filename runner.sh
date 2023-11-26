#!/bin/bash

# # To generate some dreamt data.
python run.py --multirun data.three_cameras=true data.min_blend_scale=0.1 data.brightness_multiplier_desi=1e6,1e5,1e4 data.generate=true 
python run.py data.three_cameras=false data.min_blend_scale=0.1 data.brightness_multiplier_desi=null data.generate=true 

# # Run models for coadded data
python run.py --multirun data.three_cameras=false training.live_dream=false encoder.type='fourier_coadd' training.epochs=100 training.save_every=5 data.min_blend_scale=0.1 data.brightness_multiplier_desi=null data.generate=false

# # Run models for DESI data
python run.py --multirun data.three_cameras=true training.live_dream=false encoder.type='fourier' training.epochs=100 training.save_every=5 data.min_blend_scale=0.1 data.brightness_multiplier_desi=1e6,1e5,1e4 data.generate=false

# Get results for each run
python test_results.py --multirun data.three_cameras=true training.live_dream=false encoder.type='fourier' training.epochs=100 training.save_every=5 data.min_blend_scale=0.1 data.brightness_multiplier_desi=1e6,1e5,1e4 data.generate=false
python test_results.py --multirun data.three_cameras=false training.live_dream=false encoder.type='fourier_coadd' training.epochs=100 training.save_every=5 data.min_blend_scale=0.1 data.brightness_multiplier_desi=null data.generate=false

# Make some example figures
python plot_examples.py data.three_cameras=false data.generate=false data.brightness_multiplier_desi=null
python plot_examples.py data.three_cameras=true data.generate=false data.brightness_multiplier_desi=1e5