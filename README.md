# G2-SDF
3D reconstruction of space targets based on ISAR images.

## Setting up the environment
Set up a virtual environment using: 
```
conda create -n G2SDF python=3.9
conda activate G2SDF

# Install PyTorch (choose the appropriate command based on your CUDA version)

# install other dependencies
pip install -r requirements.txt

# Install rendering module
python setup.py install
```

## Usage
1: Model initialization: Run the MATLAB code model_init.m 

2: 3D Reconstruction, run:
```
python SDF_reconstruction.py
```
3: SDF model visualization
```
python visu.py
```

## Datasets
The simulation datasets includes:
- **ISAR images**: Stored as JPG files
- **View angles**: Corresponding view angles in NPY format


## References

If you use this code, our models for your research, please cite:  
Pengling Tang, Zifei Li, Yifan Chen, Dou Sun, Feng Wang, Ya-Qiu Jin. "G2-SDF: Geometry-Guided Neural Signed Distance Field for Space Target Reconstruction."   

## Acknowledgments
* This code was originally based on the [CVPR2020-SDFDiff](https://github.com/YueJiang-nj/CVPR2020-SDFDiff). Thanks for making your code publicly available!
* mesh2sdf (License: https://github.com/wang-ps/mesh2sdf/blob/master/LISCENCE)
* ms_ssim (License: https://github.com/VainF/pytorch-msssim/blob/master/LICENSE)
