# OSClip


## Installation


```bash
# 1. Create and activate a conda environment (example with Python 3.10)
conda create -n OSClip python=3.10 -y
conda activate OSClip

# 2. Clone the repository and navigate to it
git clone https://github.com/zxk688/OSClip.git
cd OSClip

# 3. Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Data Preparation

### 1. Download Datasets
Download the benchmark datasets from public sources, for example:

- AID / UCMD / NWPU: Refer to the `GeoRSAI/SSOUDA` repository or other official links.

### 2. Organize Data Structure
Place all datasets under the `data/` folder at the root of the repository. The recommended structure is:

```
data
├── AID
│   ├── agricultural/
│   ├── baseball diamond/
│   ├── ...
│   ├── database.txt
│   ├── train.txt
│   ├── val.txt
│   └── test.txt
│
├── UCMD
│   ├── agricultural/
│   ├── baseball diamond/
│   ├── ...
│   ├── database.txt
│   ├── train.txt
│   ├── val.txt
│   └── test.txt
│
└── NWPU
    ├── agricultural/
    ├── baseball diamond/
    ├── ...
    ├── database.txt
    ├── train.txt
    ├── val.txt
    └── test.txt
```

- The `database.txt`, `train.txt`, `val.txt`, and `test.txt` files are index files. Each line typically contains the relative path to an image and its label, e.g.:  
  ```
  AID/agricultural/img_0001.jpg 0
  AID/baseball diamond/img_0002.jpg 1
  ```

---

## Training and Testing Examples

Run the following commands from the repository root:

```bash
# Phase 1: Train on AID and transfer to UCMD, NWPU (A → U and N)
python run.py --dataset_mode AID_UCMD_NWPU --phase train_uda --training_phase phase1

# Phase 2: Continue training from the best checkpoint of phase 1
python run.py --dataset_mode AID_UCMD_NWPU --phase train_uda --training_phase phase2 \
  --load_model OSClip/logs/classification_task/AID_UCMD_NWPU/checkpoints/best_model.pth

# Test: Evaluate the best model on AID → UCMD, NWPU
python run.py --dataset_mode AID_UCMD_NWPU --phase test \
  --load_model OSClip/logs/classification_task/AID_UCMD_NWPU/checkpoints/best_model.pth
```

### Common Parameters
- `--dataset_mode`: Dataset combination name (see the dataset_mode implementation in code).  
- `--phase`: Operation phase, e.g., `train_uda` for training, `test` for testing.  
- `--training_phase`: Sub-phase of training, as some workflows split training into multiple phases.  
- `--load_model`: Path to a model checkpoint (for continuing training or testing).

---

## Output Directory and Model Paths

By default, logs and checkpoints are saved under `OSClip/logs/`, for example:
```
OSClip/logs/classification_task/AID_UCMD_NWPU/checkpoints/best_model.pth
OSClip/logs/classification_task/AID_UCMD_NWPU/events.out.tfevents...
```
Paths can be modified in the code configuration (`Logger` or `opts`).

---

## FAQ & Recommendations

- Ensure CUDA, cuDNN, and GPU drivers are compatible with your PyTorch version.  
- For dependency conflicts, create a fresh conda environment with only the required packages.  
- Ensure train/val/test index files match the data loader format.  
- If training is slow or memory is limited, reduce `batch_size` or use mixed-precision training (if supported).

---

## Acknowledgements and Citation

We thank the contributors of AID, UCMD, NWPU datasets and related open-source tools.  
Please cite the datasets and repository if you use this work in publications or projects.

---

## Contact

For questions or collaborations, contact:  
**Prof. Xiaokang Zhang**  
School of Information Science and Engineering, Wuhan University of Science and Technology, China  
Email: zxk688@wust.edu.cn
