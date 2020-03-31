conda env create python=${{ matrix.python-version }} -f environment.yml
conda info --envs
eval "$(conda shell.bash hook)"
conda activate opt-mo
which python 
python -m ipykernel install --user --name opt-mo