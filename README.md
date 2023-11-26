# Simulation-Based Inference for Detecting Blends In Spectra

In a new virtual environment, install required packages. You can do this via
```shell
python3 -m venv .venv
source .venv/bin/activate
chmod +x requirements.sh
./requirements.sh
```

For the `specsim` noise model, you need to install a working version of `desimodel` in an easily accessed location. You can do this with the `install_desimodel_data` command, which should be available after running `requirements.sh` (check by running `which install_desimodel_data`). I recommend installing in this directory with
```shell
mkdir desimodel
install_desimodel_data -d ./desimodel
```

Make another directory anywhere on your device to store data, and run
```shell
export DATA_DIR=/path/to/your/data/directory
wget https://data.desi.lbl.gov/public/edr/spectro/redux/fuji/healpix/sv1/bright/103/10378/coadd-sv1-bright-10378.fits -P $DATA_DIR
```
to download early data release (EDR) data from DESI to use in some plots. 


Before running the scripts, in the ```conf/config.yaml``` file you'll need to change
```yaml
dir: /path/to/your/clone

data:
  dir: /path/to/your/data/directory
```
where `/path/to/your/data/directory` is the same data directory you just made, and `/path/to/your/clone` is the path to your clone of this repository on your machine.

```diff
- The default script below will simulate 300 GB of synthetic spectra in 
- `path/to/your/data/directory` for training the models. Please ensure 
- that you have adequate disk space, or alter the runner script.
``````

Finally, to run the scripts simply do
```shell
mkdir logs
mkdir figs
./runner.sh
```
which reproduces the figures and experimental results found in the paper. 


NB: The prior modules for the base simulator `provabgs` do not allow seeding by default. If you would like to add a seed, you can hardcode one at line 1113 in `.venv/lib/python3.10/site-packages/provabgs/infer.py` via

```python3
class Prior(object): 
    ''' base class for prior
    '''
    def __init__(self, label=None):
        self.ndim = None 
        self.ndim_sampling = None 
        self.label = label 
        self._random = np.random.mtrand.RandomState(789243)
```
where `789243` is just the default seed from the `conf/config.yaml` file. 


<!-- python -m venv .venv
source .venv/bin/activate
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
python -m pip install 'astropy @ git+https://github.com/astropy/astropy.git@v4.1'
python -m pip install 'specsim @ git+https://github.com/desihub/specsim.git@238d0b8'
python -m pip install hydra-core
python -m pip install tensorboard
python -m pip install git+https://github.com/PyWavelets/pywt.git@v1.4.1
python -m pip install hydra-joblib-launcher --upgrade

python -m pip install 'desimodel @ git+https://github.com/desihub/desimodel.git@51a58cb'
mkdir desimodel
install_desimodel_data -d ./desimodel


### Accessing DESI EDR
From the command line run,
export DATA_DIR=/path/to/your/data/directory
wget https://data.desi.lbl.gov/public/edr/spectro/redux/fuji/healpix/sv1/bright/103/10378/coadd-sv1-bright-10378.fits -P $DATA_DIR -->