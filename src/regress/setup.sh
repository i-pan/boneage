conda create -n pytorch_p37 python=3.7 pip

conda install pytorch torchvision cudatoolkit=10.0 -c pytorch
conda install -c conda-forge opencv

conda install scikit-learn scikit-image pandas pyyaml tqdm 

pip install pretrainedmodels albumentations kaggle pyarrow
pip install iterative-stratification
