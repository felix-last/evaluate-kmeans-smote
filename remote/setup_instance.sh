#!/bin/bash

# could do $ ssh -t instance@host.com << EOF
# but ain't gotta do this here, just pipe:
# cat run_remote_experiment.sh | ssh -t user@host.com

sudo apt-get --yes install python3-tk
sudo apt-get --yes install htop

git clone -b current_experiment https://github.com/felix-last/evaluate-kmeans-smote

curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
sudo python3 get-pip.py
rm get-pip.py

cd evaluate-kmeans-smote
sudo pip3 install -r requirements.txt
cd ..

mkdir results
mkdir datasets
