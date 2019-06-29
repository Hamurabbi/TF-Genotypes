Variational autoencoder for genotype data

## Overview

This script can be used to obtain a latent representation of genotype data in EIGENSTRAT format encoding 0,1 or 2 denoting the SNP loci. The key is to onehot encode the SNP loci and then flatten the input into a 3 x number_of_features vector before parsing it to the first hidden layer. The script provides the latent space means which can be thought of as a dimensionality reduction method as an alternative to classical methods such as PCA/t-SNE ect.

The script has the following arguments:

- ```--input ```: (required) path to csv file
- ```--n_feat ```: (required) number of features (SNP loci)
- ```--batch ```: batch size
- ```--n_layers ```: number of hidden layers
- ```--n_hidden ```: (required) number of hidden neurons per layer
- ```--latent ```: (required) number of hidden neurons in latent layer
- ```--epochs ```: number of training epochs


Then simply run:

```python TF-Genotypes.py --input PATH --n_feat FEATURES --batch BATCH_SIZE --n_hidden HIDDEN --latent LATENT --epochs EPOCHS ```

The script should write latent space, genotype reconstructions and observation labels to 3 files.

Example:

```python TF-Genotypes.py --input sample_genotypes.csv --n_feat 10000 --batch 50 --hidden 250 --latent 5 --epochs 20```

Then in the "vae" folder you should find the genotype reconstructions & latent space representation. 

## Notice

This method of reading entire genotype data from csv file into memory not efficient and is only for testing on small genotype sample (up to 500.000 SNP loci) and thus not intended for use on large genotype data. 
