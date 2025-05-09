# Scientific Team-Performance Evaluator

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

A graph-based framework for predicting the five-year citation trajectory of scientific papers or research teams using a temporal, heterogeneous bibliographic network.

![Model Architecture](https://via.placeholder.com/800x400?text=Model+Architecture)

---

## 1  Overview

This repository implements a **three-stage framework** that forecasts scientific impact from bibliographic data.

Two evaluation settings are supported:

1. **Paper mode** (realistic)  
   Input = topic embedding + authors + venue + references  
   Output = yearly citations of the paper (+1 … +5)

2. **Team mode** (counter-factual)  
   Input = topic embedding + authors (no venue / refs)  
   Output = yearly citations the team is expected to achieve on that topic

### Features

- **Temporal Graph Modeling**: Process yearly snapshots of bibliographic networks
- **Pre-Publication Prediction**: Generate citation curves before paper publication

---

## 2  Method

Our approach consists of three stages:

| Stage | Component | Purpose |
|-------|-----------|---------|
| ① | Temporally-aligned **R-GCN** | Encodes each yearly snapshot (papers, authors, venues, citations) with weight sharing across years. |
| ② | **Weighted imputer** | Reconstructs the would-be paper embedding in past years by mixing neighbour embeddings (authors, venue, references) with learnable scalar weights. |
| ③ | **Citation generator** | A GRU reads the 5-year sequence; three MLP heads output η, μ, σ of a log-normal curve  ![](https://latex.codecogs.com/svg.image?\widehat{C}_p^l=\alpha\bigl(\exp(\eta\Phi((\ln l-\mu)/\sigma))-1\bigr))  from which yearly counts are derived. |

Total loss  
`L = L_pred + β · L_time-smooth`  
where `L_time-smooth` discourages sudden embedding changes between
consecutive years (`β = 0` disables the term).

---

## 3  Installation

```bash
# Clone repository
git clone https://github.com/jiaweixu98/TeamingEvaluator.git
cd TeamingEvaluator

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# PyTorch – change URL for your CUDA/CPU build
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

---

## 4  Data

You can get the data here: <https://drive.google.com/drive/folders/1g7NEfuLw3pwhOUIdU_jvXnGhgI-AUFXK?usp=drive_link>

Two raw files are needed **once** to build yearly snapshots:

| File | Size | Comment |
|------|------|---------|
| `paper_nodes_GNN_yearly.json.gz` | 2 M papers / 44 k authors | metadata & citations |
| `tkg_embeddings_all_2024.npz` (768-d SPECTER2) *or* `OpenAI_paper_embeddings.npz` (256-d) | – | title/abstract embeddings |

About the data:
- The dataset is large: 2M papers and 44K authors
- For approximately 9K papers, the authors' publication lists are complete
- For simplicity, we choose the 9K papers as core papers, using them as the samples for citation labeling
- When preparing author and venue embeddings, we utilize all 2M papers' embeddings

### Auto-building yearly snapshots

Snapshots `data/raw/G_<year>.pt` (one per year) are produced
automatically by `utils/data_utils.load_snapshots`.  
To save time you can download pre-built snapshots from  
<https://drive.google.com/drive/folders/1g7NEfuLw3pwhOUIdU_jvXnGhgI-AUFXK?usp=drive_link>
and place them in `data/raw/`.

Note: When using pre-built snapshots, you don't need to rebuild the data from scratch.

---

## 5  Training

### Command-line flags (excerpt)

| Flag | Meaning | Default |
|------|---------|---------|
| `--train_years y0 y1` | inclusive range used for training | required |
| `--test_years  y0 y1` | inclusive range used for final test | required |
| `--hidden_dim`        | size of node / GRU embeddings | `50` |
| `--epochs`            | training epochs | `150` |
| `--lr`                | learning rate | `1e-3` |
| `--beta`              | temporal smoothness weight | `0.5` |
| `--cold_start_prob`   | hide venue/refs with this prob. during training | `0.3` |
| `--eval_mode`         | `paper` or `team` | `paper` |
| `--device`            | CUDA string or `cpu` | auto |

#### Smoke test

```bash
python train.py \
  --train_years 1995 1995 \
  --test_years  1996 1996 \
  --epochs 1 --hidden_dim 32 --cold_start_prob 0.5 --device cuda:0
```

#### Typical experiment

```bash
python train.py \
  --train_years 2006 2015 \
  --test_years 2016 2019 \
  --hidden_dim 64 \
  --epochs 800 \
  --cold_start_prob 0.5 \
  --beta 0.5 \
  --eval_mode team \
  --device cuda:3 \
  --load_checkpoint runs/20250507_181011_team_resumedFrom_evaluated_model_epoch020_male0_0.3730_male1_0.6771_male2_0.7959_male3_0.8540_male4_0.8855_team/evaluated_model_epoch090_male0_0.4429_male1_0.6081_male2_0.6199_male3_0.5991_male4_0.5767_team.pt
```

Logs and checkpoints appear in `runs/<timestamp>/`.

---

## 6  Inference (team mode example)

```bash
python inference.py \
  --ckpt runs/20250217_121314/best_model_epoch050_male0.4321_team.pt \
  --snapshots "data/raw/G_{}.pt" \
  --year 2020 \
  --authors 123456 789012 \
  --topic_emb path/to/topic_vec.npy \
  --device cuda:0
```

Output: predicted yearly citations for +1 … +5.

---

## 7  Baselines (ongoing)

Two reference methods (code in `baselines/`) are provided:

1. **GBM** (XGBoost) on hand-crafted features  
2. **DeepCas** skeleton ready for your random-walk & encoder implementation

Train a baseline via

```bash
python baseline_train.py \
  --model gbm \
  --train_years 2005 2015 \
  --test_years  2016 2019
```

---

## 8  Results

| Model | MALE-Y1 ↓ | MALE-Y2 ↓ | MALE-Y3 ↓ | MALE-Y4 ↓ | MALE-Y5 ↓ | MALE-Avg ↓ | RMSLE-Y1 ↓ | RMSLE-Y2 ↓ | RMSLE-Y3 ↓ | RMSLE-Y4 ↓ | RMSLE-Y5 ↓ | RMSLE-Avg ↓ |
|-------|:--------:|:--------:|:--------:|:--------:|:--------:|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|:----------:|
| GBM baseline | 0.454 | 0.649 | 0.725 | 0.758 |  0.766 | 670 | 0.615 | 0.829 | 0.923 | 0.983 | 0.981 | 0.866 |
| R-GCN + Imputer + GRU (data-leakage fixing) | 0.427 | 0.394 | 0.377 | 0.385 | 0.395 | 0.396 | 0.494 | 0.490 | 0.492 | 0.507 | 0.522 | 0.501 |
**Notes:**
- Lower values are better for all metrics (↓)
- MALE = Mean Absolute Logarithmic Error
- RMSLE = Root Mean Square Logarithmic Error
- Y1-Y5 = prediction years 1 through 5
- R-GCN results from epoch 675 (team mode), but one data lekage problem is identified, fixing it.


---

## 9  Directory Structure

```
TeamingEvaluator/
├── README.md
├── requirements.txt
│
├── data/
│   ├── raw/                ← yearly snapshots G_<year>.pt (auto-generated)
│   └── yearly_snapshots/   ← mapping tables, caches
│
├── models/
│   ├── rgcn_encoder.py
│   ├── imputer.py
│   ├── impact_rnn.py
│   └── full_model.py
│
├── utils/
│   ├── dataset_builder.py
│   ├── data_utils.py
│   └── metrics.py
│
├── train.py
└── inference.py
```