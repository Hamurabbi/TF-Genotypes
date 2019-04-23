#!/usr/bin/env python3
import argparse
import numpy as np
from tensorflow.contrib.slim import fully_connected as fc
import pandas as pd
import tensorflow as tf
import os
slim = tf.contrib.slim

"""
Variational autoencoder for genotype data.

Input is one hot encoded and flatten before given as input to the encoder function.
The mean is computed with linear activation and std with softplus activation. Then a
latent variables z is sampled from a gaussian with reparametrization trick to allow
for backpropagation. The decoder returns the flatten reconstruction which is then reshaped
and argmax to recompute the genotypes.
"""

def get_data(path):
   """Rewrite this function

   :param path: path to csv file.
   """
   df_t = pd.read_csv(path)
   df_t["person"] = df_t["Unnamed: 0"]
   df_t = df_t.drop("Unnamed: 0",1)
   return df_t

def init_warmup(warmup):
   """Set warmup
   """

   # initialize warmup
   if warmup == 0:
      beta_ = 1.0
      to_add_ = 0
   else:
      to_add_ = 1/warmup
      beta_ = 0
   beta_ = np.array([beta_,])
   return (beta_, to_add_)

def sample_z(mu, sigma):
   """
   Sample epsilon from gaussian distribution

   :param mu: latent mean
   :param sigma: latent std
   :returns: a latent sample by reparametrization trick
   """
   # Sample epsilon
   epsilon = tf.random_normal(tf.shape(sigma), name='epsilon')
   sam = mu + tf.multiply(tf.exp(0.5 * sigma), epsilon)
   return sam

def summaries(e_dict, m_dict):
   """
   Mean summaries to get per epoch values
   """
   for key, value in m_dict.items():
      e_dict[key].append(np.mean(value))
   return e_dict

def report(LOGDIR, epoch, e_dict, saver, sess, fh_log):
   """
   Report/write out after each epoch
   """
   # print loss
   print ("Epoch: %i; Loss: %f; KLd: %f; CE %f" % (epoch, e_dict["loss"][-1], e_dict["KLd"][-1], e_dict["CE"][-1]))
   fh_log.write("%i\t%0.5e\t%0.5e\t%0.5e\n" % (epoch, e_dict["loss"][-1], e_dict["KLd"][-1], e_dict["CE"][-1]))

def encoder(batch, hidden_dim, n_layers, activation,drop_rate, is_training):
   """
   encoder function

   :param batch: normalized daya batch
   :param hidden_dim: number of nodes in hidden layers
   :param n_layers: number of hidden layers
   :param activation: activation function
   :param drop_rate: dropout rate
   :param is_training: dropout during network traning
   :returns: last hidden layer
   """
   hidden_enc = []
   hidden_enc_bn = []
   for i in range(n_layers):
      if i == 0:
         hidden_enc.append(fc(batch, hidden_dim, scope="hidden_in%i" % i))
         hidden_enc_bn.append(tf.layers.batch_normalization(hidden_enc[i], name="hidden_in%i_bn" % i))
         if drop_rate > 0:
            hidden_enc_bn.append(tf.layers.dropout(hidden_enc_bn[i], rate=drop_rate, name="hiddden_in%i_dp" % i, training=is_training))
         else:
            hidden_enc_bn.append(hidden_enc_bn[i])
      else:
         hidden_enc.append(fc(hidden_enc_bn[i], hidden_dim, scope="hidden_in%i" % i))
         hidden_enc_bn.append(tf.layers.batch_normalization(hidden_enc[i], name="hidden_in%i_bn" % i))
         if drop_rate > 0:
            hidden_enc_bn.append(tf.layers.dropout(hidden_enc_bn[i+1], rate=drop_rate, name="hiddden_in%i_dp" % i, training=is_training))

   return hidden_enc_bn[-1]


def decoder(latent_var, hidden_dim, n_layers, activation,drop_rate, is_training):
   """
   decoder function

   :param latent_var: latent space sample
   :param hidden_dim: number of nodes in hidden layers
   :param n_layers: number of hidden layers
   :param activation: activation function
   :param drop_rate: dropout rate
   :param is_training: dropout during network traning
   :returns: last hidden layer
   """
   hidden_dec = []
   hidden_dec_bn = []
   for i in range(n_layers):
      if i == 0:
         hidden_dec.append(fc(latent_var, hidden_dim, scope="hidden_dec%i" % i))
         hidden_dec_bn.append(tf.layers.batch_normalization(hidden_dec[i], name="hidden_dec%i_bn" % i))
         if drop_rate > 0:
            hidden_dec_bn.append(tf.layers.dropout(hidden_dec_bn[i], rate=drop_rate, name="hiddden_dec%i_dp" % i, training=is_training))
         else:
            hidden_dec_bn.append(hidden_dec_bn[i])
      else:
         hidden_dec.append(fc(hidden_dec_bn[i], hidden_dim, scope="hidden_dec%i" % i))
         hidden_dec_bn.append(tf.layers.batch_normalization(hidden_dec[i], name="hidden_dec%i_bn" % i))
         if drop_rate > 0:
            hidden_dec_bn.append(tf.layers.dropout(hidden_dec_bn[i+1], rate=drop_rate, name="hiddden_dec%i_dp" % i, training=is_training))

   return hidden_dec_bn[-1]

def run(args):
   """ Create graph and run session """

   #Get input values
   n_feat = list(map(int, args.n_feat))[0]
   hidden = list(map(int, args.hidden))[0]
   latent = int(args.latent)
   data_path = args.i[0]
   LOGDIR = args.o
   n_layer = args.n_layers
   drop_rate = args.dp_rate

   # setup graph
   g = tf.Graph()
   with g.as_default():

      ## Setup placeholders and one hot encode input
      with tf.variable_scope('inputs', reuse=True):
         x_data = tf.placeholder(tf.float32, [None, n_feat], name="x_data")
         x_onehot = tf.one_hot(tf.cast(x_data, tf.int32),3,dtype=tf.float32)
         x_flat = tf.reshape(x_onehot, [-1, 3*n_feat])
         is_training = tf.placeholder(tf.bool)
         beta = tf.placeholder(tf.float32, [1,], name="Beta")

      #Encoder
      with tf.name_scope('encoder'):
         en = encoder(x_flat,hidden,n_layer,tf.nn.relu,drop_rate,is_training)

      #Latent layers
      with tf.name_scope('latent_space'):
         z_mean = fc(en, latent, scope='enc_fc4_mu', activation_fn=None) #  Linear activation
         z_log_sigma = fc(en, latent, scope='enc_fc4_sigma', activation_fn=tf.nn.softplus)  # softplus activation

         # Sample from gaussian distribution
         z = sample_z(z_mean, z_log_sigma)

      #Decoder
      with tf.name_scope('decoder'):
         de = decoder(z, hidden, n_layer, tf.nn.relu,drop_rate,is_training)

      # get flat reconstruction and reshape back to genotype format with argmax
      with tf.name_scope('output'):
         x_hat = fc(de, 3*n_feat, scope='dec_fc4', activation_fn=None) #linear activation
         x_hat = tf.reshape(x_hat,[-1, n_feat,3])
         x_decoded = tf.cast(tf.argmax(x_hat,axis=-1),tf.int64)

      # Loss functions
      with tf.name_scope("cross_entropy"):
         cross_entropy = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(logits=x_hat, labels=x_onehot))
         recon_loss = tf.reduce_mean(cross_entropy)
         tf.summary.scalar("cross_entropy", recon_loss)

      with tf.name_scope("KL_divergence"):
         KL_divergence = -0.5 * tf.reduce_sum(1 + z_log_sigma - tf.square(z_mean) - tf.exp(z_log_sigma), axis=1)
         latent_loss = tf.reduce_mean(KL_divergence)
         tf.summary.scalar("KL_divergence", latent_loss)

      with tf.name_scope("total_loss"):
         total_loss = tf.reduce_mean(recon_loss + tf.reduce_mean(tf.multiply(KL_divergence, beta)))
         tf.summary.scalar("total_loss", total_loss)

      # Train optimizer
      with tf.name_scope("train"):
         train_step = tf.train.AdamOptimizer(learning_rate=args.lrate).minimize(total_loss)

      # save summaries
      saver = tf.train.Saver()

      # initializer
      init = tf.global_variables_initializer()

   # prepare lists for collecting data
   epoch_dict = {"CE":[], "KLd": [], "loss": []}

   # open handle forsaving loss
   loss_file = "%s/loss.tab" % LOGDIR
   if os.path.exists(loss_file):
         os.remove(loss_file)
   fh_log = open(loss_file, "a")

   fh_log.write("Epoch\tLoss\tKL\tCE\n")

   ## Run session ##
   with tf.Session(graph=g, config=tf.ConfigProto(log_device_placement=True)) as sess:
      #sess.run()
      sess.run(init)

      beta_, to_add_ = init_warmup(args.warmup)
      merged = tf.summary.merge_all()
      train_writer = tf.summary.FileWriter(LOGDIR + '/train', sess.graph)

      ## Get genotype data
      data=get_data(data_path)
      ran=20

      # run epochs
      for epoch in range(args.nepochs):

         # prepare list for collecting data for minibatches
         mb_dict = {"CE":[], "KLd": [], "loss": []}

         # training
         data2 = data.iloc[:,0:n_feat]
         for iter in range(ran):
            D = data2.sample(n=args.batch,axis=0)
            data2=data2.drop(D.index)
            batch=np.array(D)
            _, vaecost, cre, KLd,  summary = sess.run([train_step, vae_cost, cross_entropy, KL_divergence, merged], feed_dict={x_data: batch, is_training_pl: True})
            mb_dict["loss"].append(vaecost)
            mb_dict["CE"].append(cre)
            mb_dict["KLd"].append(KLd)

         # summaries for every epoch
         epoch_dict = summaries(epoch_dict, mb_dict)

         # add to tensorboard
         train_writer.add_summary(summary, epoch)

         # after epoch add information to epoch lists and write out to file
         report(LOGDIR, epoch, epoch_dict, saver, sess, fh_log)

         # add to beta
         beta_ = beta_ + to_add_
         if beta_ > 1:
            beta_ = np.array([1,])

      # after session
      fh_log.close()

      ## get latent representation and save reconstructions
      la_dict = {"CE":[], "KLd": [], "loss": [],}

      latent_file = "%s/latent.representation.tab" % LOGDIR
      if os.path.exists(latent_file):
         os.remove(latent_file)
      open_file_1 = open(latent_file, 'ab')

      genotype_file = "%s/genotype.reconstruction.tab" % LOGDIR
      if os.path.exists(genotype_file):
         os.remove(genotype_file)
      open_file_2 = open(genotype_file, 'ab')

      labels_file = "%s/labels.reconstruction.tab" % LOGDIR
      if os.path.exists(labels_file):
         os.remove(labels_file)
      open_file_3 = open(labels_file, 'ab')

      # final pass
      for iter in range(1):
            D = data.iloc[0:,:]
            ind = D.index
            #drop those individuals after use
            data=data.drop(ind)
            batch_test=np.array(D.iloc[:,0:n_feat])
            vaecost, cre, KLd, mu, x_reconstruction = sess.run([vae_cost, cross_entropy, KL_divergence, z_mean,x_decoded], feed_dict={x_data: batch_test,beta: beta_, is_training: False})
            la_dict["loss"].append(vaecost)
            la_dict["CE"].append(cre)
            la_dict["KLd"].append(KLd)
            np.savetxt(open_file_1, mu, fmt="%.3f", delimiter="\t")
            np.savetxt(open_file_2, x_reconstruction, delimiter="\t")
            np.savetxt(open_file_3, list(ind), delimiter="\t",fmt="%s")

      # close
      open_file_1.close()
      open_file_2.close()
      open_file_3.close()

      # print loss to screen
      print ("Final model loss: %f; KLd: %f; CE %f; " % (np.mean(la_dict["loss"]), np.mean(la_dict["KLd"]), np.mean(la_dict["CE"])))

if __name__ == '__main__':

   # create the parser
   parser = argparse.ArgumentParser(prog='TF-VAE.py', formatter_class=lambda prog: argparse.HelpFormatter(prog,max_help_position=50, width=130), usage='%(prog)s [options]', description='Variational Autoencoder using TensorFlow on genotype data')
   parser.add_argument('--i', help='genotypes', required=True, nargs="+")
   parser.add_argument('--n_feat', help='number of features', nargs="+", required=True)
   parser.add_argument('--batch', help='batch size [100]', type=int, default=100)
   parser.add_argument('--n_layers', help='number of hidden layers', type=int, default=1)
   parser.add_argument('--n_hidden', help='number of hidden neurons (list)', nargs="+",required=True)
   parser.add_argument('--latent', help='number of latent neurons (list)',required=True)
   parser.add_argument('--epochs', help='number of epochs [10]', type=int, default=10)
   parser.add_argument('--lrate', help='learning rate [1e-4]', type=float, default=1e-4)
   parser.add_argument('--warmup', help='use warmup', type=int, default=0)
   parser.add_argument('--dp_rate', help='dropout rate', type=float, default=0)
   parser.add_argument('--o', help='output folder [vae]', default='vae')

   args = parser.parse_args()

   # make output dir
   if not os.path.exists(args.o):
      os.makedirs(args.o)

   run(args)


