# Composite-Degradation-Image-Restoration
NTU CCDS Final Year Project

In this project, I have made modifications to [OneRestore](https://github.com/gy65896/OneRestore) model and extended its use case to blur degradation, commonly observed in traffic scenarios. The experiment setup follows the same method used in the original paper.

## Demo

I have created a demo script "create_server.bat" that handles 
* env creation and activation
* launching the server
* opening localhost tab on browser

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

Sample images in `./image` can be used for the quick inference with our image embedder:

```
python test.py --embedder-model-path ./ckpts/embedder_model.tar --restore-model-path ./ckpts/onerestore_model.tar --input ./image/ --output ./output/ --concat
```

Input additional prompt to perform controllable restoration using our text embedder:

```
python test.py --embedder-model-path ./ckpts/embedder_model.tar --restore-model-path ./ckpts/onerestore_model --prompt low_haze --input ./image/ --output ./output/ --concat
```


## Training

### Prepare Dataset
The clear images of CDD-13 can be found in the original paper's GitHub. 
 - Generate the depth map based on [MegaDepth](https://github.com/zhengqili/MegaDepth).
 - Generate the light map based on [LIME](https://github.com/estija/LIME).
 - Generate the rain mask database based on [RainStreakGen](https://github.com/liruoteng/RainStreakGen?tab=readme-ov-file).
 - Download the snow mask database from [Snow100k](https://sites.google.com/view/yunfuliu/desnownet).
 - Blur images are generated using a blur kernel implemented with cv2.filter2D() in syn_data.py.

After doing the above steps, use [syn_data.py](https://github.com/Leozk2000/Composite-Degradation-Image-Restoration/blob/main/syn_data/syn_data.py) to generate the final isolated and composite degraded versions.


Data directory should look like this:

```
./data/
|--train
|  |--blur
|  |--blur_haze
|  |--clear
|  |  |--000001.png
|  |  |--000002.png
|  |--low
|  |--haze
|  |--rain
|  |--snow
|  |--low_haze
|  |--low_rain
|  |--low_snow
|  |--haze_rain
|  |--haze_snow
|  |--low_haze_rain
|  |--low_haze_snow
|--test
```

A generated example is as follows:

| Clear Image | Depth Map | Light Map | Rain Mask | Snow Mask
| :--- | :---| :---| :--- | :---
| <img src="https://github.com/gy65896/OneRestore/blob/main/img_file/clear_img.jpg" width="200"> | <img src="https://github.com/gy65896/OneRestore/blob/main/img_file/low.jpg" width="200"> | <img src="https://github.com/gy65896/OneRestore/blob/main/img_file/light_map.jpg" width="200"> | <img src="https://github.com/gy65896/OneRestore/blob/main/img_file/rain_mask.jpg" width="200"> | <img src="https://github.com/gy65896/OneRestore/blob/main/img_file/snow_mask.png" width="200">