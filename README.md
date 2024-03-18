<div id="top"></div>
<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Don't forget to give the project a star!
*** Thanks again! Now go create something AMAZING! :D
-->




[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]



<!-- PROJECT LOGO -->
<br />
<div align="center">
<!--
  <a href="https://github.com/matteo-bastico/MI-Seg">
    <img src="images/logo.png" alt="Logo" width="80" height="80">
  </a>-->

<h3 align="center">MI-Seg</h3>

  <p align="center">
    MI-Seg is a framework based on <a href="https://github.com/Project-MONAI/MONAI">MONAI</a> libray for Cross-Modality 
clinical images Segmentation using Conditional Models and Interleaved Training.  
    <br />
    <a href="https://github.com/matteo-bastico/MI-Seg"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <!--
    <a href="https://github.com/matteo-bastico/MI-Seg">View Demo</a>
    · -->
    <a href="https://github.com/matteo-bastico/MI-Seg/issues">Report Bug</a>
    ·
    <a href="https://github.com/matteo-bastico/MI-Seg/issues">Request Feature</a>
  </p>
</div>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#citation">Citation</a></li>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a>
      <ul>
        <li><a href="#dataset">Dataset</a></li>
        <li><a href="#training">Training</a></li>
        <li><a href="#testing">Testing</a></li>
      </ul>
    </li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->
<!--## About The Project
Describe project here.-->
<!--
<p align="center">
    <img height=300px src="images/architecture.png">
</p>
-->
### Citation

Our paper has been accepted at ICCVW 2023 and is available [here](https://openaccess.thecvf.com/content/ICCV2023W/LXCV/papers/Bastico_A_Simple_and_Robust_Framework_for_Cross-Modality_Medical_Image_Segmentation_ICCVW_2023_paper.pdf) and on [ArXiv](https://arxiv.org/abs/2310.05572). Please cite our work with
```sh
  @InProceedings{Bastico_2023_ICCV,
    author    = {Bastico, Matteo and Ryckelynck, David and Cort\'e, Laurent and Tillier, Yannick and Decenci\`ere, Etienne},
    title     = {A Simple and Robust Framework for Cross-Modality Medical Image Segmentation Applied to Vision Transformers},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV) Workshops},
    month     = {October},
    year      = {2023},
    pages     = {4128-4138}
  }
  ```

<p align="right">(<a href="#top">back to top</a>)</p>

### Built With
Our released implementation is tested on:
* Ubuntu 22.04
* Python 3.10.8
* PyTorch 1.13.1 and PyTorch Lightning 1.8.6
* Ray 2.2.0
* NVIDIA CUDA 11.7
* Monai 1.1.0
* Optuna 3.1.0

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- GETTING STARTED -->
## Getting Started

### Prerequisites

* Clone our project folder
* Create and lunch your conda environment with
  ```sh
  conda create -n MI-Seg python=3.10.8
  conda activate MI-Seg
  ```
<!--### Installation-->
* Install dependencies
    ```sh
  pip install -r requirements.txt
  ```
  Note: for Pytorch CUDA installation follow https://pytorch.org/get-started/locally/.
  
<p align="right">(<a href="#top">back to top</a>)</p>



<!-- USAGE EXAMPLES -->
## Usage

### Dataset
The dataset used in our experiments can be downloaded [here](https://zmiclab.github.io/zxh/0/mmwhs/) upon access request.
Download  and unzip it into `/dataset/MM-WHS` folder.

[Optional] Convert label and Perform N4 Bias Correction of MRIs using the provided Notebook load_data.ipynb

You should end up with a similar data structure (sub-folders are not represented here)
```sh
  MM-WHS
  ├── ct_train	# Ct training folder
  │   ├── ct_train_1001_image.nii.gz # Image
  │   ├── ct_train_1001_label.nii.gz # Label
  │   ...
  ├── ct_test
  ├── mr_train
  ├── mr_test
  ...
  ```

The splits we used for our cross_validation are provided in `CT_fold1.json` and `CT_fold2.json`.
### Training
To train a model you can use the `train.py` script provided. Single training are based on PyTorch Lightning and 
all the Trainer arguments can be passed to the script 
(see [here](https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html)). Additionally, we provide model, 
data and logger-specific arguments. To have a full list of the possible arguments execute `python train.py --help`.

An example of C-Swin-UNETR training on single GPU is shown in the following
```
python train.py --model_name=swin_unetr --out_channels=6 --feature_size=48 --num_heads=3 --accelerator=gpu --devices=1 --max_epochs=2500 --encoder_norm_name=instance_cond --vit_norm_name=instance_cond --lr=1e-4 --batch_size=1 --patches_training_sample=1
``` 

The available models are unet, unetr and swin_unetr and pre_swin_unetr (in this case the [pretrained model](https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/model_swinvit.pt) of monai
must be provided as `--pre_swin`. 

Furthermore, we use [WandB](https://wandb.ai/site) to log the experiments and specifications can be set as arguments. 
In the previous example wandb will run in online mode, so you need to provided login and API key. To change wandb mode set 
`wandb_mode=offline`.

`Note:` AMP (`--no_amp`) should be disabled with checkpointing to save memory during training of Swin_Unetr based models (`--use_checkpoint`).

### Testing 
Our pre-trained models can be downloaded [here](https://drive.google.com/drive/folders/1S-GdryfNziYV4088jh4meVbqdWkdEwuL?usp=sharing) and tested with the `test.py` script. The path of the model weights 
should be provided as `--checkpoint` (note that the model weight should be under the `state_dict` key). 

Example:

```
python test.py --out_channels=6 --model_name=swin_unetr --num_workers=2 --feature_size=48  --num_heads=3 --encoder_norm_name=instance_cond --vit_norm_name=instance_cond --checkpoint=experiments/<path>
```

### Hyper-parameters Optimization

Hyper-parameters optimization is based on [Optuna](https://optuna.org/). For the moment, the script supports automatic setup of distributed 
tuning ONLY on Slurm environments. Therefore, it needs to be adapted by the user to run in different multi-GPUs enviroments.

The hyper-parameters grid is set in automatic for each model as stated in our paper and the tuning can be started as in the following. 
The script will run 10 trials, with TPE optimizer and ASHA pruner, and save the in the `MI-Seg.log` log file (if Slurm) or `MI-Seg.sqlite` (if not Slurm).

```
python -u tune.py --num_workers=2 --out_channels=6 --no_include_background --criterion=generalized_dice_focal --scheduler=warmup_cosine --model_name=swin_unetr --n_trials=10 --study_name=c-swin-unetr --max_epochs=2500 --check_val_every_n_epoch=50 --batch_size=1 --patches_training_sample=4 --iters_to_accumulate=4 --cycles=0.5 --storage_name=MI-Seg --min_lr=1e-5 --max_lr=1e-3 --vit_norm_name=instance_cond --encoder_norm_name=instance_cond  --port=23456
```

The script can be run multiple time with the same `--storage_name` in order to continue a previous tuning.

To open log files dashboards not stored as RDB, we provide the `utils/run_server.py --path=<storage>` script.
The dashboard of our tuning presented in the paper is available at `experiments/optuna/MI-Seg.log` and can be open with

```
python utils/run_server.py --path=experiments/optuna/MI-Seg.log
```

<p align="right">(<a href="#top">back to top</a>)</p>

### Pre-Trained Models

The best pre-trained model weights for Conditional UNet and Swin-UNETR resulting from our hyper-parameters optimization 
can be downloaded [here](https://drive.google.com/drive/folders/1S-GdryfNziYV4088jh4meVbqdWkdEwuL?usp=sharing). 

For instance, to produce the segmentation on the test dataset using the provided weights you can run for Conditional UNet:

```
python predict_whs.py --model=unet_vanilla --encoder_norm_name=instance_cond --feature_size 16 64 128 256 512 --num_res_units=3 --strides 1 2 2 2 1 --out_channels=8 --checkpoint=path/to/weights.pt --result_dir=path/to/result
```

or for Conditional Swin-UNETR:

```
python -u predict_whs.py --model=swin_unetr --encoder_norm_name=instance_cond --vit_norm_name=instance_cond --feature_size=36 --num_heads=4 --out_channels=8 --checkpoint=path/to/weights.pt --result_dir=path/to/result
```

<!-- ROADMAP -->
## Roadmap

- [ ] Implement LN for convolutional layers of Monai (testing purposes)
- [ ] Implement distributed tuning on not-Slurm environment
<!--
- [ ] Feature 2
- [ ] Feature 3
    - [ ] Nested Feature-->

See the [open issues](https://github.com/matteo-bastico/MI-Seg/issues) for a full list of proposed features (and known issues).

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- CONTRIBUTING -->
## Contributing

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/my_feature`)
3. Commit your Changes (`git commit -m 'Add my_feature'`)
4. Push to the Branch (`git push origin feature/my_feature`)
5. Open a Pull Request

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- LICENSE -->
## License

Distributed under the MIT (or other) License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- CONTACT -->
## Contact

Matteo Bastico - [@matteobastico](https://twitter.com/matteobastico) - matteo.bastico@minesparis.psl.eu

Project Link: [https://github.com/matteo-bastico/MI-Seg](https://github.com/matteo-bastico/MI-Seg)

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

This  work  was  supported  by  the  H2020  European  Project ...

<p align="right">(<a href="#top">back to top</a>)</p>


<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/matteo-bastico/MI-Seg.svg?style=for-the-badge
[contributors-url]: https://github.com/matteo-bastico/MI-Seg/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/matteo-bastico/MI-Seg.svg?style=for-the-badge
[forks-url]: https://github.com/matteo-bastico/MI-Seg/network/members
[stars-shield]: https://img.shields.io/github/stars/matteo-bastico/MI-Seg.svg?style=for-the-badge
[stars-url]: https://github.com/matteo-bastico/MI-Seg/stargazers
[issues-shield]: https://img.shields.io/github/issues/matteo-bastico/MI-Seg.svg?style=for-the-badge
[issues-url]: https://github.com/matteo-bastico/MI-Seg/issues
[license-shield]: https://img.shields.io/github/license/matteo-bastico/MI-Seg.svg?style=for-the-badge
[license-url]: https://github.com/matteo-bastico/MI-Seg/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://www.linkedin.com/in/matteo-bastico/
