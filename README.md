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
    MI-Seg is a framework based on <a href="https://github.com/Project-MONAI/MONAI">MONAI</a> libray for Modality 
Independent clinical images Segmentation using Conditional Models.  
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
## About The Project
Describe project here.
<!--
<p align="center">
    <img height=300px src="images/architecture.png">
</p>
-->
### Citation

Our paper is available in [example](https://www.example.com). Please cite our work with
```sh
  @article{
    
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
The dataset used in our experiments can be downloaded [here](https://zenodo.org/record/583096#.Y8fNVOzMKV4) upon access request.
Download  and unzip it into `/MultiModalPelvic/data` folder before applying our pre-processing provided.

Execute the pre-processing scripts in the following order:
<ul>
<li> prepare_data.py: in order to group files into folder corresponding to image modalities</li>
<li> compress_data.py: generates compressed .nii.gz files for MRI and Deformed CT volumes</li>
<li> convert_labels.py: converts RT Structs files into .nii.gz files containing the labels</li>
<li> [Optional] Perform N4 Bias Correction of T2-Weighted MRIs using bias_correction.py</li>
</ul> 

You should end up with a similar data structure (sub-folders are not represented here)
```sh
  MultiModalPelvic
  ├── data	
  │   ├── 1_01_P # Patient Folder
  │   │   ...
  │   │   ├── Deformed_CT.nii.gz
  │   │   └── labels.nii.gz
  │   │   └── T2.nii.gz
  │   │   └── T2_N4.nii.gz
  │   ...
  ```
### Training
 

### Testing 
Our pre-trained models can be downloaded [here](drive)

### Hyper-parameters Optimization


<p align="right">(<a href="#top">back to top</a>)</p>



<!-- ROADMAP -->
## Roadmap

- [ ] Add here future steps
- [ ] Step 2
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
