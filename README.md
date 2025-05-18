# Composite-Degradation-Image-Restoration
NTU CCDS Final Year Project

In this project, I have made modifications to [OneRestore](https://github.com/gy65896/OneRestore) model and extended its use case to blur degradation, commonly observed in traffic scenarios. The experiment setup follows the same method used in the original paper.

## Demo

I have created a demo script "create_server.bat" that handles 
1) env creation and activation
2) launching the server
3) opening localhost tab on browser
Minimum requirement is to have anaconda installed. [Environment.yml](https://github.com/Leozk2000/Composite-Degradation-Image-Restoration/blob/main/environment.yml) includes cuda-enabled pytorch so its environment creation may fail if you do not have a Nvidia GPU.


## Testing

### Environment Creation

If you have not run the demo steps and wish to manually create the environment, please run the following steps.

Install
- python 3.10
- cuda 11.7

```
# git clone this repository
git clone https://github.com/Leozk2000/Composite-Degradation-Image-Restoration.git
cd Composite-Degradation-Image-Restoration

# create new anaconda env
conda env create -f environment.yml

# activate anaconda env
conda activate onerestore
```

### Pretrained Models
Both embedder and model checkpoints can be found in [ckpts](https://github.com/Leozk2000/Composite-Degradation-Image-Restoration/tree/main/ckpts)

### Inference

Sample images in `./image` can be used for the quick inference:

```
python test.py --embedder-model-path ./ckpts/embedder_model.tar --restore-model-path ./ckpts/onerestore_model.tar --input ./image/ --output ./output/ --concat
```

You can also input the prompt to perform controllable restoration. For example:

```
python test.py --embedder-model-path ./ckpts/embedder_model.tar --restore-model-path ./ckpts/onerestore_model --prompt low_haze --input ./image/ --output ./output/ --concat
```


## Training

### Prepare Dataset
The clear, base images of CDD-11/13 can be found in the original paper's GitHub. The respective degraded images can be generated using syn_data.py found within [syn_data]{https://github.com/Leozk2000/Composite-Degradation-Image-Restoration/blob/main/syn_data/syn_data.py} folder.