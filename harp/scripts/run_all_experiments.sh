#!/bin/bash

# make it all useable
chmod +x harp/scripts/data.sh
chmod +x harp/scripts/evaluate.sh
chmod +x harp/scripts/train.sh

# get data
harp/scripts/data.sh --config harp/config/data_config.yaml

# train all networks
harp/scripts/train.sh --config harp/config/f_train_config.yaml
hap/scripts/train.sh --config harp/config/csc_g_train_config.yaml
harp/scripts/train.sh --config harp/config/harp_pi_train_config.yaml

# evaluate all networks
harp/scripts/evaluate.sh --config harp/config/evaluate_config.yaml