## Repository Manual

### Prerequisites

Install Libraries
```bash
pip install -r requirements.txt
```

Download Dataset

```bash
cd ComAlign
wget http://images.cocodataset.org/zips/val2017.zip
unzip val2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip annotations_trainval2017.zip
```


### 1. **Experiment Pipelines**


```
python main.py --space experiment --task bias --gpath <unzipped_dataset_path> --opath <output_path>
```

