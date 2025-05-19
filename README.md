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

Isolated:

```
python syn_data.py --hq-file ./data/clear/ --light-file ./data/light_map/ --out-file ./out/ --low 
```

Composite:

```
python syn_data.py --hq-file ./data/clear/ --light-file ./data/light_map/ --depth-file ./data/depth_map/ --rain-file ./data/rain_mask/ --snow-file ./data/snow_mask/ --out-file ./out/ --low --haze --rain
```


A generated example is as follows:

| Clear Image | Low Light | Haze | Rain | Snow | Blur
| :--- | :---| :---| :--- | :--- | :---
| <img src="https://github.com/Leozk2000/Composite-Degradation-Image-Restoration/blob/main/img_file/clear_img.jpg" width="200"> | <img src="https://github.com/Leozk2000/Composite-Degradation-Image-Restoration/blob/main/img_file/lowlight.png" width="200"> | <img src="https://github.com/Leozk2000/Composite-Degradation-Image-Restoration/blob/main/img_file/haze.png" width="200"> | <img src="https://github.com/Leozk2000/Composite-Degradation-Image-Restoration/blob/main/img_file/rain.png" width="200"> | <img src="https://github.com/Leozk2000/Composite-Degradation-Image-Restoration/blob/main/img_file/snow.png" width="200"> | <img src="https://github.com/Leozk2000/Composite-Degradation-Image-Restoration/blob/main/img_file/blur.png" width="200">


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


### Train Model

**1. Train Text/Visual Embedder by**

```
python train_Embedder.py --train-dir ./data/CDD-13_train --test-dir ./data/CDD-13_test --check-dir ./ckpts --batch 256 --num-workers 0 --epoch 200 --lr 1e-4 --lr-decay 50
```

**2. Remove the optimizer weights in the Embedder model file by**

```
python remove_optim.py --type Embedder --input-file ./ckpts/embedder_model.tar --output-file ./ckpts/embedder_model.tar
```

**3. Generate the `dataset.h5` file for training OneRestore by**

```
python makedataset.py --train-path ./data/CDD-13_train --data-name dataset.h5 --patch-size 256 --stride 200
```

**4. Train OneRestore model by**

```
python train_OneRestore_single-gpu.py --embedder-model-path ./ckpts/embedder_model.tar --save-model-path ./ckpts --train-input ./dataset.h5 --test-input ./data/CDD-11_test --output ./result/ --epoch 170 --bs 4 --lr 2e-4 --adjust-lr 20 --num-works 4
```

**5. Remove the optimizer weights in the OneRestore model file by**

```
python remove_optim.py --type OneRestore --input-file ./ckpts/onerestore_model.tar --output-file ./ckpts/onerestore_model.tar
```


## Performance

### CDD-11

</div>
<div align=center>
<img src="https://github.com/Leozk2000/Composite-Degradation-Image-Restoration/blob/main/img_file/Table5.1.png" width="1080">
<img src="https://github.com/Leozk2000/Composite-Degradation-Image-Restoration/blob/main/img_file/Table5.3.png" width="1080">
</div>

### CDD-13

</div>
<div align=center>
<img src="https://github.com/Leozk2000/Composite-Degradation-Image-Restoration/blob/main/img_file/Table5.2.png" width="1080">
<img src="https://github.com/Leozk2000/Composite-Degradation-Image-Restoration/blob/main/img_file/Table5.4.png" width="1080">
</div>

### Blur-related degradations

</div>
<div align=center>
<img src="https://github.com/Leozk2000/Composite-Degradation-Image-Restoration/blob/main/img_file/Table5.5.png" width="1080">
<img src="https://github.com/Leozk2000/Composite-Degradation-Image-Restoration/blob/main/img_file/comparison_grid.png" width="1080">
</div>

### Real Benchmark Datasets

</div>
<div align=center>
<img src="https://github.com/Leozk2000/Composite-Degradation-Image-Restoration/blob/main/img_file/Table5.6.png" width="1080">
</div>

### Real Traffic Scenes

</div>
<div align=center>
<img src="https://github.com/Leozk2000/Composite-Degradation-Image-Restoration/blob/main/img_file/realscenes1.png" width="1080">
<img src="https://github.com/Leozk2000/Composite-Degradation-Image-Restoration/blob/main/img_file/realscenes2.png" width="1080">
</div>


### Ablation Study

</div>
<div align=center>
<img src="https://github.com/Leozk2000/Composite-Degradation-Image-Restoration/blob/main/img_file/ablation study.png" width="1080">
</div>


## Citation

```
@misc{leo2025composite,
  author       = {Leo, Zhi Kai},
  title        = {Composite Degradation Image Restoration},
  year         = {2025},
  note         = {Final Year Project (FYP), Nanyang Technological University, Singapore},
  howpublished = {\url{https://hdl.handle.net/10356/184085}}
}
```
