# DISC: Dual-Imbalance Synergistic Calibration for Graph Anomaly Detection
----------------------------------------------------------
## Overview

This repository provides the implementation of **DISC**, a semi-supervised graph anomaly detection framework designed for scenarios with limited labeled data.

DISC focuses on two imbalance problems in self-training-based graph anomaly detection. First, the labeled training set usually contains substantially fewer anomalous nodes than normal nodes, which may result in an initially biased decision boundary. Second, high-confidence pseudo-labels generated during self-training tend to be dominated by normal nodes, causing the prediction bias to accumulate over successive training iterations.

To address these problems, DISC introduces focal loss for supervised calibration and DAR loss for correcting the class-frequency imbalance and label noise in the high-confidence pseudo-label set.

## Requirements

The experiments were conducted with the following environment:

```text
Python 3.11.14
PyTorch 2.4.0
```

The main Python dependencies include:

```text
dgl
numpy
pandas
scikit-learn
```

## Repository Structure

```text
DISC/
├── config/
│   ├── yelp.yml
│   └── reddit.yml
├── data/
├── modules/
├── main_DISC.py
├── models.py
└── README.md
```

## Datasets

The experiments are conducted on two real-world graph anomaly detection datasets: **YelpChi** and **Reddit**.

### YelpChi

YelpChi can be obtained through the DGL fraud dataset interface:

```text
https://docs.dgl.ai/en/0.8.x/api/python/dgl.data.html
```

### Reddit

The Reddit dataset source is provided at:

```text
https://github.com/mala-lab/Awesome-Deep-Graph-Anomaly-Detection
```

Please download the datasets and place the corresponding files in the `data/` 

## Data Split

For each dataset, 1% of the nodes are randomly selected as the labeled training set. The remaining nodes are divided into the validation and test sets at a ratio of 1:2.

The validation set is used for model selection.

## Running the Code

Run DISC on YelpChi using:

```bash
python main_DISC.py --config config/yelp.yml
```

Run DISC on Reddit using:

```bash
python main_DISC.py --config config/reddit.yml
```

## Key Hyperparameters

| Parameter | YelpChi | Reddit | Description |
|---|---:|---:|---|
| `epochs` | 200 | 200 | Number of training epochs |
| `lr` | 0.001 | 0.001 | Learning rate |
| `training-ratio` | 1 | 1 | Percentage of labeled training nodes |
| `batch-size` | 128 | 32 | Batch size |
| `normal-th` | 5 | 5 | Confidence threshold for normal pseudo-labels |
| `fraud-th` | 85 | 85 | Confidence threshold for anomalous pseudo-labels |
| `unsup-weight` | 0.7 | 0.7 | Weight of the unsupervised loss |
| α                 |     0.7 |    0.7 | Class Balancing Factor |
| λ                 |     0.5 |    0.7 | hyperparameter of DAR loss in the whole objective |

## Evaluation Metrics

The model is evaluated using the following metrics:

- AUROC
- AUPRC
- Macro-F1
