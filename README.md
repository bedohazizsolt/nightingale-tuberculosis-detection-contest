Nightingale Detecting Active Tuberculosis Bacilli - 2024
===========

This repository contains the winning solution for the [Detecting Active Tuberculosis Bacilli - 2024 Contest](https://app.nightingalescience.org/contests/m3rl61qq21wo) hosted by Nightingale Open Science and Wellgen Medical.

### *[Zsolt Bedőházi](https://www.linkedin.com/in/zsoltbedohazi/), [András M. Biricz](https://github.com/abiricz)*

## How our approach works?

Our proposed methodology for TB detection utilizes a weakly-supervised framework, effectively addressing data privacy and accessibility challenges by eliminating the dependency on external datasets and annotations made by experts. Employing transfer learning, we leverage state-of-the-art pre-trained vision encoders to extract relevant features from TB images in the absence of direct annotations. Subsequently, these features are utilized to train various self-supervised models (Transformers and MIL), enabling them to learn directly from the data using only image-level labels. Our approach provides a rapid, scalable, and efficient solution for TB detection from sputum microscopy images, improving diagnostic capabilities in resource-limited areas.

## Usage

<details>
<summary>
Extract patches from images
</summary>

```bash
01_raw_patches/
  └──extract_patches_and_coords.py
 ```
</details>


<details>
<summary>
Generate patch embeddings
</summary>

```bash
02_patch_embeddings/
  ├── generate_dinov2-vitlarge_embeddings.py
  ├── generate_dinov2-vitsmall_embeddings.py
  └── generate_uni_embeddings.py
 ```
</details>


<details>
<summary>
Training
</summary>

```bash
03_training/
  └──hipt_stage3_on_embeddings_bag/
      └── hipt_stage3_training-balanced-folds-cross-val-uni-efficientloader-scheduler.ipynb
  └──transformer_on_embeddings_bag/
      ├── train_transformer_cls_on_embeddings_bag.py
      └── train_transformer_cls_on_embeddings_bag_multi_branch.py
 ```
</details>


<details>
<summary>
Prediction
</summary>

```bash
04_prediction/
  └──hipt_stage3_on_embeddings_bag/
      └── predict_tb_with_hipt_stage3_dinov2-vit_10fold_improved.ipynb
  └──transformer_on_embeddings_bag/
      ├── predict_tb_with_transformer_dinov2-vit_10fold_improved.ipynb
      └── predict_tb_with_transformer_dinov2-vit-multibranch_10fold_improved.ipynb
  └──predict_tb_fusion.ipynb
 ```
</details>

## Pre-trained models

<table>
  <tr>
    <th>Model</th>
    <th>Arch</th>
    <th># of params</th>
    <th># of train images</th>
    <th>Download</th>
  </tr>
  
  <tr>
    <td>dino2-small</td>
    <td>ViT-S/14 distilled</td>
    <td align="center">21 M</td>
    <td align="center">142 M</td>
    <td align="center"><a href="https://huggingface.co/facebook/dinov2-small">link</a></td>
  </tr>
  
  <tr>
    <td>dinov2-large</td>
    <td>ViT-L/14 distilled</td>
    <td align="center">300 M</td>
    <td align="center">142 M</td>
    <td align="center"><a href="https://huggingface.co/facebook/dinov2-large">link</a></td>
  </tr>
  
  <tr>
    <td>UNI</td>
    <td>ViT-L/14 distilled</td>
    <td align="center">300 M</td>
    <td align="center">100 M</td>
    <td align="center"><a href="https://huggingface.co/MahmoodLab/UNI">link</a></td>
  </tr>
</table>

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
