#!/bin/bash

cd evaluate-kmeans-smote
python3 remote.py _dataset
tar -xfzv ~/datasets/dataset.tar.gz ~/datasets
