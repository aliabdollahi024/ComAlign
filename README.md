## ComAlign

### Prerequisites

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

 Model             | Checkpoint Link                                |
|-------------------|------------------------------------------------|
| **CLIP-VIT-B32**   | [Link to ComAlign-CLIP-VIT-B32 Checkpoint](https://drive.google.com/uc?id=1QxCig_UP5fCORMeCeNRtrycktZHXQOvS)        |


```
python inference.py 
```

