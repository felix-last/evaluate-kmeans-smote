#!/bin/bash

cd evaluate-kmeans-smote
python3 remote.py _dataset
cd ..
cd datasets
tar -xzvf ~/datasets/dataset.tar.gz ~/datasets
