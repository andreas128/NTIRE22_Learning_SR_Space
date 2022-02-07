# Learning the Super-Resolution Space Challenge

Develop a super-resolution method that can actively sample from the space of plausible super-resolutions.

We are part of the CVPR 2022 [NTIRE](https://data.vision.ee.ethz.ch/cvl/ntire22/) New Trends in Image Restoration and Enhancement workshop.

## How to participate?

To participate in this challenge, please sign up using the following link and clone this repo to benchmark your results.

<p align="center">
  <a href="https://bit.ly/34E0rLB">
  <img width="140px" alt="CVPR 2022 Challenge Signup" src="https://user-images.githubusercontent.com/11280511/105862009-9d8cb980-5fef-11eb-8952-3e29b628afb7.png">
  </a>
</p>

Challenge participants can submit their **paper to this CVPR 2022 Workshop.**

## Tackling the ill-posed nature of Super-Resolution
[![CVPR 2022 Challenge](https://user-images.githubusercontent.com/11280511/152827880-e6e2d831-be32-4225-bdd6-2edabaf08ac9.gif)
](https://bit.ly/34E0rLB)

Usually, super-resolution (SR) is trained using pairs of high- and low-resolution images. **Infinitely many high-resolution images can be downsampled to the same low-resolution image.** That means that the problem is ill-posed and cannot be inverted with a deterministic mapping. Instead, one can frame the SR problem as learning a stochastic mapping, capable of **sampling from the space of plausible high-resolution images given a low-resolution image**. This problem has been addressed in recent works [1, 2, 3]. The one-to-many stochastic formulation of the SR problem allows for a few potential advantages:
* The development of more robust learning formulations that better accounts for the ill-posed nature of the SR problem.
* Multiple predictions can be sampled and compared.  
* It opens the potential for controllable exploration and editing in the space of SR predictions.   

| <img width="1000" alt="Super-Resolution with Normalizing Flow" src="https://user-images.githubusercontent.com/11280511/104035941-152aae00-51d3-11eb-9294-6fc71489c562.png"> | <img width="1000" alt="Explorable SR" src="https://user-images.githubusercontent.com/11280511/104035744-c7ae4100-51d2-11eb-9e1c-e501020c9216.png">  | <img width="1000" alt="Screenshot 2021-01-12 at 16 05 43" src="https://user-images.githubusercontent.com/11280511/104332087-1a983900-54f0-11eb-8b69-0656eaaa6c84.png"> |
| :--: | :--: | :--: |
| [[Paper]](http://de.arxiv.org/pdf/2006.14200) [[Project]](https://github.com/andreas128/SRFlow) | [[Paper]](https://arxiv.org/pdf/1912.01839.pdf) [[Project]](https://github.com/YuvalBahat/Explorable-Super-Resolution) | [[Paper]](https://arxiv.org/pdf/2004.04433.pdf) [[Project]](https://mcbuehler.github.io/DeepSEE/) |
| [1]  SRFlow: Learning the Super-Resolution Space with Normalizing Flow. Lugmayr et al., ECCV 2020.  | [2]  Explorable Super-Resolution. Bahat & Michaeli, CVPR 2020.  | [3] DeepSEE: Deep Disentangled Semantic Explorative Extreme Super-Resolution. Bühler et al., ACCV 2020.  |



## CVPR 2022 Challenge on Learning the Super-Resolution Space

We organize this challenge to stimulate research in the emerging area of learning one-to-many SR mappings that are capable of sampling from the space of plausible solutions. Therefore the task is to develop a super-resolution method that: 
1. Each individual SR prediction should achieve highest possible **photo-realism**, as perceived by humans.
2. Is capable of sampling an arbitrary number of SR images capturing **meaningful diversity**, corresponding to the *uncertainty* induced by the ill-posed nature of the SR problem together with image priors.
3. Each individual SR prediction should be consistent with the input low-resolution image.

The challenge contains two tracks, targeting 4X and 8X super-resolution respectively. You can download the training and validation data in the table below. At a later stage, the low-resolution of the test set will be released.

<table>
<thead>
<tr>
<th>&nbsp;</th>
<th colspan="2">Training</th>
<th colspan="2">Validation</th>
</tr>
</thead>
<tbody>
<tr>
<td>&nbsp;</td>
<td>Low-Resolution</td>
<td>High-Resolution</td>
<td>Low-Resolution</td>
<td>High-Resolution</td>
</tr>
<tr>
<td>Track 4X</td>
<td><a href="https://data.vision.ee.ethz.ch/alugmayr/NTIRE2021/DIV2K-tr_4X.zip" rel="nofollow">4X LR Train</a></td>
<td><a href="https://data.vision.ee.ethz.ch/alugmayr/NTIRE2021/DIV2K-tr_1X.zip" rel="nofollow">4X HR Train</a></td>
<td><a href="https://data.vision.ee.ethz.ch/alugmayr/NTIRE2021/DIV2K-va_4X.zip" rel="nofollow">4X LR Valid</a></td>
<td><a href="https://data.vision.ee.ethz.ch/alugmayr/NTIRE2021/DIV2K-va_1X.zip" rel="nofollow">4X HR Valid</a></td>
</tr>
<tr>
<td>Track 8X</td>
<td><a href="https://data.vision.ee.ethz.ch/alugmayr/NTIRE2021/DIV2K-tr_8X.zip" rel="nofollow">8X LR Train</a></td>
<td><a href="https://data.vision.ee.ethz.ch/alugmayr/NTIRE2021/DIV2K-tr_1X.zip" rel="nofollow">8X HR Train</a></td>
<td><a href="https://data.vision.ee.ethz.ch/alugmayr/NTIRE2021/DIV2K-va_8X.zip" rel="nofollow">8X LR Valid</a></td>
<td><a href="https://data.vision.ee.ethz.ch/alugmayr/NTIRE2021/DIV2K-va_1X.zip" rel="nofollow">8X HR Valid</a></td>
</tr>
</tbody>
</table>

## Results from Challenge on Learning the Super-Resolution Space 2021

[[Paper](https://bit.ly/3Lf5F0P)] Full Report

![2021_CVPR_NTIRE_SRSpace_gif](https://user-images.githubusercontent.com/11280511/152773008-39df8217-fb8d-4d43-99d1-45d4525c44c0.gif)


### Winner Team: Deepest

[[Paper]](https://openaccess.thecvf.com/content/CVPR2021W/NTIRE/papers/Kim_Noise_Conditional_Flow_Model_for_Learning_the_Super-Resolution_Space_CVPRW_2021_paper.pdf) [[Code]](https://github.com/younggeun-kim/NCSR)
Noise Conditional Flow Model for Learning the Super-Resolution Space
- Combine SRFlow with SoftFlow
- Models Noise

 
### Runner-Up (8x) Team: CLIPLAB

[[Paper]](https://openaccess.thecvf.com/content/CVPR2021W/NTIRE/papers/Jo_SRFlow-DA_Super-Resolution_Using_Normalizing_Flow_With_Deep_Convolutional_Block_CVPRW_2021_paper.pdf) [[Code]](https://github.com/yhjo09/SRFlow-DA)
SRFlow-DA: Super-Resolution Using Normalizing Flow with Deep
- Based on SRFlow
- New LR encoder

 
### Runner-Up (4x) Team: njtech&seu

- Based on SRFlow
- Improved LR encoding
- Improved Affine couplings


### IMLE Method: FutureReference
[[Paper]](https://arxiv.org/pdf/2011.01926.pdf) [[Code]](https://github.com/niopeng/HyperRIM/tree/main/code) Generating Unobserved Alternatives

- Based on Implicit Maximum Likelihood Estimation
- Uses progressive Upscaling

### Most Diverse VAE Method: SR_DL
[[Paper]](https://openaccess.thecvf.com/content_CVPRW_2020/papers/w31/Liu_Unsupervised_Real_Image_Super-Resolution_via_Generative_Variational_AutoEncoder_CVPRW_2020_paper.pdf) Variational AutoEncoder for Reference based Image Super-Resolution

- Reference based
- Components: VGG Encoder, Conditional Variational AutoEncoder, Image Decoder

### Most Diverse GAN Method: SSS
Flexible SR using Conditional Objective

- Based on RRDB and SFT
- Components: SR-, Conditioning Branch

## Challenge Rules

To guide the research towards useful and generalizable techniques, submissions need to adhere to the following rules. All participants must submit code of their solution along with the final results.

* The method must be able to generate an **arbitrary number** of diverse samples. That is, your method cannot be limited to a maximum number of different SR samples (corresponding to e.g. a certain number of different output network heads).
* All SR samples must be generated by a **single model**. That is, **no ensembles** are allowed.
* **No self-ensembles** during inference (e.g. flipping and rotation).
* All SR samples must be generated using the **same hyper-parameters**. That is, the generated SR samples shall not be the result of different choices of hyper-parameters during inference.
* We accept submissions of **deterministic methods**. However, they will naturally score zero in the diversity measure and therefore **not** be able to win the challenge.
* Other than the validation and test split of the DIV2k dataset, **any training data** or pre-training is allowed. You are not allowed to use DIV2K validation or test sets (low- and high-resolution images) for training.


## Evaluation Protocol

A method is evaluated by first predicting a **set of 10** randomly sampled SR images for each low-resolution image in the dataset. From this set of images, evaluation metrics corresponding to the three criteria above will be considered. The participating methods will be ranked according to each metric. These ranks will then be combined into a final score. The three evaluation metrics are described next.


```bash
git clone --recursive https://github.com/andreas128/NTIRE22_Learning_SR_Space.git
python3 measure.py OutName path/to/Ground-Truch path/to/Super-Resolution n_samples scale_factor

# n_samples = 10
# scale_factor = 4 for 4X and 8 for 8X
```

### How we measure Photo-realism?
To assess the photo-realism, a **human study** will be performed on the test set for the final submission.

Automatically assessing the photo-realism and image quality is an extremely difficult task. All existing methods have severe shortcomings. As a very rough guide, you can use the LPIPS distance. **Note:** LPIPS will not be used to score photo-realism of you final submission. So beware of overfitting to LPIPS, as that can lead to worse results. LPIPS is integrated in our provided [toolkit](#evaluation-protocol) in `measure.py`.

### How we measure the spanning of the SR Space?
The samples of the developed method should provide a meaningful diversity. To measure that, we define the following score. We sample 10 images, densely calculate a metric between the samples and the ground truth. To obtain the *local best* we pixel-wise select the best score out of the 10 samples and take the full image's average. The *global best* is obtained by averaging the whole image's score and selecting the best. Finally, we calculate the score using the following formula:

score = (global best - local best)/(global best) * 100


|  | [ESRGAN](https://github.com/xinntao/ESRGAN) | [SRFlow](https://github.com/andreas128/SRFlow) |
| :--: | :--: | :--: |
| Track 4X | 0 | 25.36 |
| Track 8X | 0 | 10.62 |


### How we measure the Low Resolution Consistency
To measure how much information is preserved in the super-resloved image from the low-resolution image, we measure the LR-PSNR. The goal in this challenge is to obtain a LR-PSNR of 45dB. All approaches that have an average PSNR above this value will be ranked equally in terms of this criteria.


|  | [ESRGAN](https://github.com/xinntao/ESRGAN) | [SRFlow](https://github.com/andreas128/SRFlow) |
| :--: | :--: | :--: |
| Track 4X | 39.01 | 49.91 |
| Track 8X | 31.28 | 50.0  |



## Important Dates

| Date       | Event |
| ---------- | ----  |
| 2022.03.13 | Final test data release (inputs only) |
| 2022.03.20 | test result submission deadline |
| 2022.03.20 | fact sheet / code / model submission deadline |
| 2022.04.01 | challenge paper submission deadline |
| 2022.04.08 | camera-ready deadline |
| 2022.06.19 | workshop day |


## Submission of Final Test Results

After the final testing phase, participants will be asked to submit:
* SR predictions on the test set
* A fact sheet describing their method
* Code

**Sign up to get notified about further details.**


## Issues and questions
In case of any questions about the challenge or the toolkit, feel free to open an issue on Github.

## Organizers
* [Andreas Lugmayr](https://twitter.com/AndreasLugmayr) (andreas.lugmayr@vision.ee.ethz.ch)
* [Martin Danelljan](https://martin-danelljan.github.io/) (martin.danelljan@vision.ee.ethz.ch)
* [Radu Timofte](http://people.ee.ethz.ch/~timofter/) (radu.timofte@vision.ee.ethz.ch)

The terms and conditions for participating in the challenge are provided [here](TERMSandCONDITIONS.md)


## How to participate?

To participate in this challenge, please sign up using the following link and clone this repo to benchmark your results.

<p align="center">
  <a href="https://bit.ly/34E0rLB">
  <img width="140px" alt="CVPR 2022 Challenge Signup" src="https://user-images.githubusercontent.com/11280511/105862009-9d8cb980-5fef-11eb-8952-3e29b628afb7.png">
  </a>
</p>

Challenge participants can submit their **paper to this CVPR 2022 Workshop.**
