#!/bin/bash
python -m pip install torch
python -m pip install einops
python -m pip install torchmetrics
python -m pip install SciencePlots
python -m pip install fitsio
python -m pip install astroquery
python -m pip install 'provabgs @ git+https://github.com/changhoonhahn/provabgs.git'
python -m pip install 'desiutil @ git+https://github.com/desihub/desiutil.git'
python -m pip install 'desispec @ git+https://github.com/desihub/desispec.git'
python -m pip install 'desitarget @ git+https://github.com/desihub/desitarget.git'
python -m pip install 'specsim @ git+https://github.com/desihub/specsim.git@238d0b8'
python -m pip install hydra-core
python -m pip install git+https://github.com/PyWavelets/pywt.git@v1.4.1
python -m pip install hydra-joblib-launcher --upgrade
python -m pip install 'desimodel @ git+https://github.com/desihub/desimodel.git@51a58cb'
