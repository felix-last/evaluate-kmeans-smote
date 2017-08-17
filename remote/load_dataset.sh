#!/bin/bash

cd evaluate-kmeans-smote
python3 remote.py _dataset
tar -xzvf ~/datasets/dataset.tar.gz ~/datasets
