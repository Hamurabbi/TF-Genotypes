import numpy as np
import pandas as pd
from tensorflow.keras.layers import Lambda, Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.losses import mse, binary_crossentropy
from tensorflow.keras.utils import plot_model
from tensorflow.keras import backend as K
import tensorflow as tf
import argparse
import os
import matplotlib.pyplot as plt

def plot_results(encoder,
                 data,
                 batch_size=128,
                 model_name="vae_genotype"):
    """
    Plot the latent space mean
    """

    x_test = data
    os.makedirs("results", exist_ok=True)

    filename = os.path.join("results", "vae_mean.png")
    z_mean, _, _ = encoder.predict(x_test,
                                   batch_size=batch_size)
    plt.figure(figsize=(12, 10))
    plt.scatter(z_mean[:, 0], z_mean[:, 1], c='k')
    plt.colorbar()
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.savefig(filename)


def sampling(args):
    """
    Reparameterization trick by sampling from an isotropic unit Gaussian.
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean = 0 and std = 1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

def preprocess(data, shape):
    """This function creates one hot encoding of genotypes"""
    encoded = tf.keras.utils.to_categorical(data)
    flatten=tf.keras.backend.reshape(encoded,(-1,shape))
    return tf.keras.backend.eval(flatten)

def get_data(path):
	"""
	Get csv genotype data, create one hot encoding and 
	return train and test set
	"""
	data = pd.read_csv(path, index_col=0)
	df_test = data.sample(frac=0.20)
	df_train= data.drop(df_test.index, axis=0)

	X_train_hot = preprocess(df_train.iloc[:,0:10000], 30000)
	X_test_hot = preprocess(df_test.iloc[:,0:10000], 30000)
	return X_train_hot, X_test_hot



def vae_model(input_shape,
			hidden_dim,
			latent_dim,
			original_dim):
	"""
	Variational autoencoder

	:param input_shape: flatten genotype input shape
	:param hidden_dim: dimension of hidden layer
	:param latent_dim: dimension of latent layer
	:param original_dim: dimension of original genotype data
	"""

	inputs = Input(shape=input_shape, name='encoder_input')
	x = Dense(hidden_dim, activation='relu')(inputs)

	z_mean = Dense(latent_dim, name='z_mean')(x)
	z_log_var = Dense(latent_dim, name='z_log_var')(x)
	z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

	# encoder model
	encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
	latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
	encoder.summary()

	x = Dense(hidden_dim, activation='relu')(latent_inputs)
	outputs = Dense(original_dim, activation='softmax')(x)

	# decoder model
	decoder = Model(latent_inputs, outputs, name='decoder')
	decoder.summary()

	outputs = decoder(encoder(inputs)[2])
	vae = Model(inputs, outputs, name='vae_mlp')

	# define loss
	# reconstruction loss
	reconstruction_loss = binary_crossentropy(inputs, outputs)
	reconstruction_loss *= original_dim

	# kl divergence
	kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
	kl_loss = K.sum(kl_loss, axis=-1)
	kl_loss *= -0.5
	vae_loss = K.mean(reconstruction_loss + kl_loss)
	
	vae.add_loss(vae_loss)
	vae.compile(optimizer='adam')
	vae.summary()

	return vae, encoder

if __name__ == '__main__':

	parser = argparse.ArgumentParser(prog='TF-Genotypes.py', formatter_class=lambda prog: argparse.HelpFormatter(prog,max_help_position=50, width=130), usage='%(prog)s [options]', description='Variational Autoencoder using TensorFlow on genotype data')
	parser.add_argument('--input', help='genotypes', required=True, nargs="+")
	parser.add_argument('--n_feat', help='number of features', nargs="+", required=True, type=int)
	parser.add_argument('--batch', help='batch size [100]', type=int, default=100)
	parser.add_argument('--hidden', help='number of hidden neurons',required=True, type=int)
	parser.add_argument('--latent', help='number of latent neurons',required=True, type=int)
	parser.add_argument('--epochs', help='number of epochs [10]', type=int, default=10)
	args = parser.parse_args()
	
	original_dim = args.n_feat[0]*3
	input_shape = (args.n_feat[0]*3, )
	intermediate_dim = args.hidden
	batch_size = args.batch
	latent_dim = args.latent
	epochs = args.epochs

	X_train, X_test = get_data(args.input[0])
	vae, encoder_model = vae_model(input_shape, intermediate_dim, latent_dim, original_dim )

	#train
	vae.fit(X_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, None), verbose=1)

	# Plot latent space
	plot_results(encoder_model, X_test, batch_size=batch_size, model_name="lanent_space")