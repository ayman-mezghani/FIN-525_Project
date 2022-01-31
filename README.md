# Financial Big Data Project: Clustering
Maxence Robaux, Ayman Mezghani


## Data
* [Link](https://drive.google.com/file/d/1-wFsDaSplHvlUinjZpHhXypNJ9CAhnLK/view?usp=sharing) 

Data is to be extracted in the following manner:
```
├── clean/
│   ├── 1m/
│   ├── 2m/
│   ├── 5m/
│   └── 60m/
├── GSPC.parquet
└── raw/
    ├── 1m/
    ├── 2m/
    ├── 5m/
    └── 60m/
```

## Files
* `clustering.py`: script file, contains the clustering code
* `utils.py`: script file, contains plotting helpers
* `data_process.ipynb` notebook, allows to compute log returns and process raw data
* `Introduction.ipynb`: notebook, contains the code used to produce introduction plots
* `Cluster_assets.ipynb`: notebook, contains the code used to cluster by assets
* `Cluster_date_5m.ipynb`: notebook, contains the code used to cluster 5m data by time
* `Cluster_date_60m.ipynb`: notebook, contains the code used to cluster 60m data by time
