# Teaching GAN to generate per-pixel annotation
This repository contains the mxnet implementation for the method described in the paper "Teaching GAN to generate per-pixel annotation".

### Installation

- Clone this repo and install dependencies:
```bash
git clone https://github.com/author-hidden-name/GAN-segmentation
cd GAN-segmentation
pip3 install -r requirements.txt
```

- Download [stylegan-models](https://drive.google.com/open?id=1vP2zUZ9NSJDFy2cc8b2mGp7MINyPW46X) converted to mxnet and unzip archive to `stylegan-models` directory.

- Download [annotated samples](https://drive.google.com/open?id=143dRAyJcRDqygepSz8lIr8ElAnwF3xp_) and unzip archive to `experiments` directory to reproduce experiment with FFHQ hair segmentation.

Be sure that your project structure is 
```
    .
    ├── deeplabv3plus
    ├── experiments
    │   ├── ffhq-hair
    │   |   ├── checkpoints
    |   |   ├── data
    |   |   └── dataset
    ├── stylegan-models
    │   ├── stylegan-bedrooms.params
    │   ├── stylegan-cars.params
    |   └── stylegan-ffhq.params
    └── ...
```

### FFHQ hair segmentation experiment 

#### Prepare config file
```bash
cp config.yml.example config.yml
```
You can specify directory with experiment using `BASE_DIR` parameter, by default it set to `experiments/ffhq-hair`.

#### Train the Decoder
Train the Decoder using annotated samples from `BASE_DIR/data`
```bash
python3 main.py train
```
The decoder weights will be saved to `experiments/ffhq-hair/checkpoints/checkpoint_last.params`

#### Generate synthetic dataset
```bash
python3 main.py generate
```
This will create `experiments/ffhq-hair/dataset/train_generated` directory with generated fake images and synthetic annotation. By default 10000 samples are created.

#### Train and test DeepLabV3+ model on generated samples
Train:
```bash
cd deeplabv3plus/experiments/rgb_segmentation/01_hair_deeplabv3_ffhq_pretrain_gan
python3 main.py train --batch-size=4 --gpus=0 --test-batch-size=4 --workers=4 --kvstore=nccl --input-path "../../../../experiments/ffhq-hair/dataset"
```
This will create subdirectory in `01_hair_deeplabv3_ffhq_pretrain_gan/runs` with logs and checkpoints

Test:
```bash
python3 main.py test --batch-size=8 --gpus=0,1 --workers=4 --kvstore=local runs/<run_subdirectory>
```


### Interactive annotator

You can 
