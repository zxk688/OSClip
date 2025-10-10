## Installation
```
Set up conda envirnment:

conda create -n OSClip
conda activate OSClip

git clone https://github.com/zxk688/OSClip.git
cd OSClip

pip install -r requirements.txt
```

## Data Preparation
1. Download Dataset
* [AID, UCMD, NWPU](https://github.com/GeoRSAI/PCLUDA)


2. Prepare File Structure

* Please put all the datasets under the ```data```. The prepared directory ```data``` would look like:
```
  data
  ├── AID
  │   ├── agricultural/
  │   ├── baseball diamond/
  │   ├── ...
  │   ├── database.txt
  │   ├── test.txt
  │   ├── train.txt
  │   ├── val.txt
  ├── UCMD
  │   ├── agricultural/
  │   ├── baseball diamond/
  │   ├── ...
  │   ├── database.txt
  │   ├── test.txt
  │   ├── train.txt
  │   ├── val.txt
  ├── NWPU
  │   ├── agricultural/
  │   ├── baseball diamond/
  │   ├── ...
  │   ├── database.txt
  │   ├── test.txt
  │   ├── train.txt
  │   ├── val.txt

```

## Training and Testing 
The training and testing script examples are as follows:
```
# train on AID → UCMD, NWPU (A → U and N)
python run.py --dataset_mode AID_UCMD_NWPU --phase train_uda --training_phase phase1 
python run.py --dataset_mode AID_UCMD_NWPU --phase train_uda --training_phase phase2 --load_model OSClip/logs/classification_task/AID_UCMD_NWPU/checkpoints/best_model.pth
# test on AID → UCMD, NWPU (A → U and N)
python run.py --dataset_mode AID_UCMD_NWPU --phase test --load_model OSClip/logs/classification_task/AID_UCMD_NWPU/checkpoints/best_model.pth 
```
