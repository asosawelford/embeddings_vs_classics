embeddings_vs_classics/
│
├── data/
│   ├── metadata/
│   │   ├── train_metadata.csv
│   │   ├── validation_metadata.csv
│   │   └── test_metadata.csv
│   │
│   └── features/
│       ├── explainable/  (e.g., P001_task1.csv, P002_task1.csv, ...)
│       └── embeddings/   (e.g., P001_task1.npy, P002_task1.npy, ...)
│
├── src/
│   ├── dataloader.py     # Our custom Dataset class
│   ├── models.py         # Definitions for our 3 ANNs
│   ├── train.py          # The main training and evaluation script
│   └── utils.py          # Helper functions (e.g., for metrics)
│
└── notebooks/
    └── 01_data_exploration.ipynb

