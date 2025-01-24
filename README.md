## Repository Manual

### Prerequisites

Install Libraries
```bash
pip install -r requirements.txt
```

Download Data

```bash
cd ComAlign
wget http://images.cocodataset.org/zips/val2017.zip
unzip val2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
gdown https://drive.google.com/uc?id=1vloi1qL85bM8yNgRQsRWyIX1gMHD4tj-
gdown https://drive.google.com/uc?id=1LJiC0LznQBBMu-hbM2Edmzdul22u2E7y
gdown https://drive.google.com/uc?id=1QxCig_UP5fCORMeCeNRtrycktZHXQOvS
unzip embeddings.zip
```


### 1. **Experiment Pipelines**


```
python main.py --space experiment --task bias --gpath <unzipped_dataset_path> --opath <output_path>
```

