Variational autoencoder for genotype data

## Overview

This script can be used to obtain a latent representation & reconstructions of genotype data in EIGENSTRAT format encoding 0,1 or 2 denoting the SNP loci. The script has the following arguments:

- ```--i ```: (required) path to csv file
- ```--n_feat ```: (required) number of features (SNP loci)
- ```--batch ```: batch size
- ```--n_layers ```: number of hidden layers
- ```--n_hidden ```: (required) number of hidden neurons per layer
- ```--latent ```: (required) number of hidden neurons in latent layer
- ```--epochs ```: number of training epochs
- ```--lrate ```: learning rate in gradient descent
- ```--warmup ```: how many warmup epochs
- ```--dp_rate ```: dropout rate in each hidden layer
- ```--o ```: output folder

Then simply run:

```python TF-VAE.py --i PATH --n_feat FEATURES --n_hidden HIDDEN --latent LATENT ```

The script should write latent space, genotype reconstructions and observation labels to 3 files.

## Notice

This method of reading entire genotype data from csv file into memory not efficient and is only for testing on small genotype sample (up to 500.000 SNP loci) and thus not intended for use on large genotype data. 
