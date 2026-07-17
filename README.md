# DISC
----------------------------------------------------------
Python 3.11.14 PyTorch 2.4.0

For YelpChi please refer to https://docs.dgl.ai/en/0.8.x/api/python/dgl.data.html.

For Reddit, please refer to https://github.com/mala-lab/Awesome-Deep-Graph-Anomaly-Detection.

From each dataset, 1% of the nodes are randomly selected as the training set, and the remaining nodes are split in a 1:2 ratio into validation and test sets.

# YelpChi
python main_DISC.py --config config/yelp.yml

# Reddit
python main_DISC.py --config config/reddit.yml

| Arguments                | YelpChi | Reddit |
| ----------------- | ------: | -----: |
| epochs            |     200 |    200 | 
| training-ratio    |       1 |      1 |
| batch-size        |     128 |     32 | 
| normal-th         |       5 |      5 | 
| fraud-th          |      85 |     85 | 
| α                 |     0.7 |    0.7 | 
| λ                 |     0.5 |    0.7 | 
