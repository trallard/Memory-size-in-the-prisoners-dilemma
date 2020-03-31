conda env create -f environment.yml
conda info --envs
eval "$(conda shell.bash hook)"
conda activate opt-mo
python -m ipykernel install --user --name opt-mo