# ComAlign: Compositional Alignment in Vision-Language Models

This repository accompanies the research on the Gender-Activity Binding (GAB) bias in Vision-Language Models (VLMs). The GAB bias refers to the tendency of VLMs to incorrectly associate certain activities with a gender based on ingrained stereotypes. This research introduces the GAB dataset, comprising approximately 5,500 AI-generated images depicting various activities performed by individuals of different genders. The dataset is designed to assess and quantify the extent of gender bias in VLMs, particularly in text-to-image and image-to-text retrieval tasks.

Our experiments reveal that VLMs experience a significant drop in performance when the gender of the person performing an activity does not align with stereotypical expectations. Specifically, the presence of an unexpected gender performing a stereotyped activity leads to an average performance decline of about 13.2% in image-to-text retrieval tasks. Additionally, when both genders are present in the scene, the models are often biased toward associating the activity with the gender expected to perform it. The study also explores the bias in text encoders and their role in the gender-activity binding phenomenon.

Below is an overview of the creation process of the GAB dataset and the empirical tests conducted to assess the gender-activity binding bias:

![Main Figure](./image.png)

In this repository, we provide the code and dataset (GABDataset) used to examine gender bias in Vision-Language Models (VLMs) through various experiments described in the main paper. The repository is organized into three phases: **phaze1**, **phaze2**, and **phaze3**, with corresponding directories for each experiment.


## Repository Manual

Install Libraries
```bash
git clone https://github.com/aliabdollahi024/ComAlign.git
cd ComAlign
pip install -r requirements.txt
```

Download Data

```bash
wget http://images.cocodataset.org/zips/val2017.zip
unzip val2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
gdown https://drive.google.com/uc?id=1vloi1qL85bM8yNgRQsRWyIX1gMHD4tj-
gdown https://drive.google.com/uc?id=1LJiC0LznQBBMu-hbM2Edmzdul22u2E7y
gdown https://drive.google.com/uc?id=1QxCig_UP5fCORMeCeNRtrycktZHXQOvS
unzip embeddings.zip
```


### 1. **Experiments**

#### 1.2. **Retrieval Experiment**

In this version, we have only provided the checkpoint for CLIP-VIT-B32. In future updates, we will gradually introduce the checkpoints and embedding files for other base VLMs.

 Model             | Checkpoint Link                                | COCO-Val Embeddings Link                                |
|-------------------|------------------------------------------------|------------------------------------------------|
| **CLIP-VIT-B32**   | [Link to ComAlign-CLIP-VIT-B32 Checkpoint](https://drive.google.com/uc?id=1QxCig_UP5fCORMeCeNRtrycktZHXQOvS)        | [Link to COCO-Val embedded by CLIP-VIT-B32](https://drive.google.com/uc?id=1vloi1qL85bM8yNgRQsRWyIX1gMHD4tj-)        |


```
python inference.py 
```

