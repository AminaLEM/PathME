#Amina LEMSARA
#CONSTANTINE 2  UNIVERSITY
#2019
#
# ==============================================================================
import tensorflow as tf
import pickle
import math
import numpy as np
from scipy import stats
import csv
import time
from sklearn.model_selection import KFold
from hyperopt import hp
from hyperopt import fmin, tpe, Trials,STATUS_OK
from sklearn import model_selection
import hyperopt
import random
import matplotlib.pyplot as plt
import shap
from scipy.spatial import distance
class MutiViewAutoencoder():
    '''
      This is the implementation of the Multi-View autoencoder
    '''
    def __init__(self,data1, data2, data3, n_hiddensh=1, activation=tf.nn.relu):

        # training datasets
        self.training_data1 = data1
        self.training_data2 = data2
        self.training_data3 = data3


        # number of features
        self.n_input1=data1.shape[1]
        self.n_input2=data2.shape[1]
        self.n_input3=data3.shape[1]

        self.n_hiddensh=n_hiddensh
        self.activation = activation



#------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------
    def encode(self,X1,X2,X3):

# =============================================================================
#         first hidden layer composed of three parts related to three sources
#          - build a layer
#          - apply the batch normalization
#          - apply a non-liner activation function
# =============================================================================

        l1= tf.layers.dense(X1, self.n_hidden1, kernel_initializer=self._init, name= 'layer1')
        l1 = tf.nn.dropout(l1, self.keep_prob)
        l1 = tf.layers.batch_normalization(l1,training=self.is_train)
        l1 = self.activation(l1)


        l2= tf.layers.dense(X2, self.n_hidden2, kernel_initializer=self._init, name= 'layer2')
        l2 = tf.nn.dropout(l2, self.keep_prob)        
        l2 = tf.layers.batch_normalization(l2,training=self.is_train)
        l2 = self.activation(l2)



        l3= tf.layers.dense(X3, self.n_hidden3, kernel_initializer=self._init,
                                 name= 'layer3')
        l3 = tf.nn.dropout(l3, self.keep_prob)
        l3 = tf.layers.batch_normalization(l3,training=self.is_train)
        l3 = self.activation(l3)

# =============================================================================
# fuse the parts of the first hidden alyer
# =============================================================================
        l= tf.layers.dense(tf.concat([l1,l2,l3],1), self.n_hiddensh, kernel_initializer=self._init,
                                name= 'layer4')
        l = tf.layers.batch_normalization(l,training=self.is_train)
        l = self.activation(l)

        return l

#------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------
    def decode(self,H):

        l= tf.layers.dense(H, self.n_hidden1 +self.n_hidden2 +self.n_hidden3, kernel_initializer=self._init,
                                 name= 'layer5')
        l = tf.layers.batch_normalization(l,training=self.is_train)
        l = self.activation(l)

        s1,s2,s3 = tf.split(l,[self.n_hidden1,self.n_hidden2,self.n_hidden3],1)

        l1= tf.layers.dense(s1, self.n_input1, kernel_initializer=self._init,
                                 name= 'layer6')
        l1 = tf.layers.batch_normalization(l1,training=self.is_train)
        l1 = self.activation(l1)

        l2= tf.layers.dense(s2, self.n_input2, kernel_initializer=self._init,
                                 name= 'layer7')
        l2 = tf.layers.batch_normalization(l2,training=self.is_train)
        l2 = self.activation(l2)

        l3= tf.layers.dense(s3, self.n_input3, kernel_initializer=self._init,
                                 name= 'layer8')
        l3 = tf.layers.batch_normalization(l3,training=self.is_train)
        l3 = self.activation(l3)


        return l1,l2,l3
#------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------

    def get_weights(self):

        with tf.variable_scope("layer1", reuse=True):
            self.W1 = tf.get_variable("kernel")
        with tf.variable_scope("layer2", reuse=True):
            self.W2 = tf.get_variable("kernel")
        with tf.variable_scope("layer3", reuse=True):
            self.W3 = tf.get_variable("kernel")
        with tf.variable_scope("layer4", reuse=True):
            self.Wsh = tf.get_variable("kernel")
        with tf.variable_scope("layer5", reuse=True):
            self.Wsht = tf.get_variable("kernel")
        with tf.variable_scope("layer6", reuse=True):
            self.W1t = tf.get_variable("kernel")
        with tf.variable_scope("layer7", reuse=True):
            self.W2t = tf.get_variable("kernel")
        with tf.variable_scope("layer8", reuse=True):
            self.W3t = tf.get_variable("kernel")
#------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------

    def L1regularization(self,weights):
        return tf.reduce_sum(tf.abs(weights))

    def L2regularization(self,weights,nbunits):
        return  math.sqrt(nbunits)*tf.nn.l2_loss(weights)
#------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------
    def loss(self,X1,Y1,X2,Y2,X3,Y3):

        self.H = self.encode(X1,X2,X3)
        X1_,X2_,X3_=self.decode(self.H)
        self.get_weights()

        # Sparse group lasso
        sgroup_lasso = self.L2regularization(self.W1,self.n_input1* self.n_hidden1) + self.L2regularization(self.W2,self.n_input2*self.n_hidden2) + self.L2regularization(self.W3,self.n_input3*self.n_hidden3)

        #Lasso
        lasso = self.L1regularization(self.W1) + self.L1regularization(self.W2) + self.L1regularization(self.W3) \
                       +self.L1regularization(self.Wsh)+ self.L1regularization(self.Wsht)\
                       + self.L1regularization(self.W1t) + self.L1regularization(self.W2t) + self.L1regularization(self.W3t)
        #Reconstruction Error
        error = tf.losses.mean_squared_error(Y1,X1_) +tf.losses.mean_squared_error(Y2,X2_) +tf.losses.mean_squared_error(Y3,X3_)
        # Loss function
        cost= 0.5*error+ 0.5*self.lamda*(1-self.alpha)*sgroup_lasso+ 0.5*self.lamda*self.alpha*lasso

        return cost


    def corrupt(self, input_data):

        noisy_input = input_data + .2 * np.random.random_sample((input_data.shape)) - .1
        output = input_data
        return noisy_input,output
#------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------
    def Normalize(self, df, mean=None, std= None):
#        if mean is None:
#            mean = np.mean(df, axis=0)
#        if std is None:
#            std = np.std(df, axis=0,ddof=0)
#        array= np.transpose((np.transpose(df) - mean.reshape((df.shape[1],1)))/std.reshape((df.shape[1],1)))

    # Scale to [0,1]
        scaled_input_1 = np.divide((df-df.min()), (df.max()-df.min()))
#    # Scale to [-1,1]
        array = (scaled_input_1*2)-1

        return array
#------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------
    def train(self):
# =============================================================================
        # training data
            train_input1 = self.Normalize(self.input1)
            train_output1 = self.Normalize(self.output1)

            train_input2 = self.Normalize(self.input2)
            train_output2 = self.Normalize(self.output2)

            train_input3 = self.Normalize(self.input3)
            train_output3 = self.Normalize(self.output3)
# =============================================================================
# =============================================================================
            save_sess=self.sess

# =============================================================================
#           costs history :
            costs = []

            costs_inter=[]
# =============================================================================
# =============================================================================
#           for early stopping :
            best_cost=100000
            stop = False
            last_improvement=0
# =============================================================================

            n_samples = train_input1.shape[0] # size of the training set

# =============================================================================
#           train the mini_batches model using the early stopping criteria
            epoch = 0
            while epoch < self.max_epochs and stop == False:
#                train the model on the traning set by mini batches
#                   suffle then split the training set to mini-batches of size self.batch_size
                seq =list(range(n_samples))
                random.shuffle(seq)
                mini_batches = [
                    seq[k:k+self.batch_size]
                    for k in range(0,n_samples, self.batch_size)
                ]

                avg_cost = 0. # The average cost of mini_batches

                for sample in mini_batches:

                    batch_xs1 = train_input1[sample][:]
                    batch_ys1 =train_output1[sample][:]

                    batch_xs2 = train_input2[sample][:]
                    batch_ys2 = train_output2[sample][:]

                    batch_xs3 = train_input3[sample][:]
                    batch_ys3 = train_output3[sample][:]

                    feed_dictio={self.X1: batch_xs1,self.Y1:batch_ys1,self.X2: batch_xs2,self.Y2:batch_ys2,self.X3: batch_xs3,self.Y3:batch_ys3, self.is_train:True, self.keep_prob:self.kp }
                    cost=self.sess.run([self.loss_,self.train_step], feed_dict=feed_dictio)
                    avg_cost += cost[0] *len(sample)/n_samples



#               cost history since the last best cost
                costs_inter.append(avg_cost)

                #early stopping based on the validation set/ max_steps_without_decrease of the loss value : require_improvement
                if avg_cost < best_cost:
                    save_sess= self.sess # save session
                    best_cost = avg_cost
                    costs +=costs_inter # costs history of the validatio set
                    last_improvement = 0
                    costs_inter= []
                else:
                    last_improvement +=1
                if last_improvement > self.require_improvement:
#                   #  print("No improvement found during the ( self.require_improvement) last iterations, stopping optimization.")
                    # Break out from the loop.
                     stop = True
                     self.sess=save_sess # restore session with the best cost

                epoch +=1

# =====================================End of model training ========================================
            feed_dictio={self.X1: train_input1,self.Y1:train_output1,self.X2: train_input2,self.Y2:train_output2,self.X3: train_input3,self.Y3:train_output3, self.is_train:False , self.keep_prob:1}
            costfinal, res=self.sess.run([self.loss_,self.H], feed_dict=feed_dictio)

#                normalize costs history
#            costs_val = (costs_val-min(costs_val) ) / (max(costs_val)-min(costs_val))
#            costs = (costs-min(costs) ) / (max(costs)-min(costs))
# =============================================================================
#            Display loss
         #   print(epoch)
            plt.plot(costs)
            plt.ylabel('cost_test_data') ,;
            plt.xlabel('iterations')
            plt.title("Learning rate =" + str(self.learning_rate))
            plt.show()
                   
# =============================================================================
           # print(best_cost)
         #   print(costfinal)
           # res = self.sess.run(self.H, feed_dict=feed_dictio)

            return costfinal, res
#------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------
    def cross_validation(self, params):

        #retrieve parameters
        self.batch_size = params['batch_size']
        self.n_hidden1=params['units1']
        self.n_hidden2=params['units2']
        self.n_hidden3=params['units3']
        self.alpha=params['alpha']
        self.lamda=params['lamda']
        self.kp=params['keep_prob']
        self.learning_rate=params['learning_rate']
        self.require_improvement= 20
        self.max_epochs = 1000
        init = params['initializer']
        if init == 'normal':
            self._init = tf.random_normal_initializer
        if init == 'uniform':
            self._init=tf.random_uniform_initializer
        if init == 'He':
            self._init = tf.contrib.layers.variance_scaling_initializer()
        if init == 'xavier':
            self._init = tf.contrib.layers.xavier_initializer()

        opt = params['optimizer']
        if opt == 'SGD':
            self.optimizer=tf.train.GradientDescentOptimizer
        if opt == 'adam':
            self.optimizer=tf.train.AdamOptimizer
        if opt == 'nadam':
            self.optimizer=tf.contrib.opt.NadamOptimizer
        if opt == 'Momentum':
            self.optimizer=tf.train.MomentumOptimizer
        if opt == 'RMSProp':
            self.optimizer=tf.train.RMSPropOptimizer
        k=5
        print("H.layer1:",self.n_hidden1, ", H.layer1:", self.n_hidden2, ", H.layer1:",self.n_hidden3)
        print( "k",k,"lamda", self.lamda, ", batch_size:",self.batch_size,", alpha:", self.alpha,", learning_rate:",self.learning_rate )
        print("initializer: ",init,'optimizer:', opt, ' ,kp:', self.kp)

        # add corruption to the traning set

        self.input1,self.output1 = self.corrupt(self.training_data1)
        self.input2,self.output2 = self.corrupt(self.training_data2)
        self.input3,self.output3 = self.corrupt(self.training_data3)

        # cross-validation
       
        loss=0

            #reset tensor graph after each cross_validation run
        tf.reset_default_graph()
            
            
        self.X1=tf.placeholder("float",shape=[None,self.training_data1.shape[1]])
        self.Y1=tf.placeholder("float",shape=[None,self.training_data1.shape[1]])

        self.X2=tf.placeholder("float",shape=[None,self.training_data2.shape[1]])
        self.Y2=tf.placeholder("float",shape=[None,self.training_data2.shape[1]])

        self.X3=tf.placeholder("float",shape=[None,self.training_data3.shape[1]])
        self.Y3=tf.placeholder("float",shape=[None,self.training_data3.shape[1]])
        self.is_train = tf.placeholder(tf.bool, name="is_train");
        self.keep_prob = tf.placeholder(tf.float32)

        self.loss_=self.loss(self.X1,self.Y1,self.X2,self.Y2,self.X3,self.Y3)
        if opt == 'Momentum':
                self.train_step = self.optimizer(self.learning_rate,0.9).minimize(self.loss_)
        else:
                self.train_step = self.optimizer(self.learning_rate).minimize(self.loss_)
            # Initiate a tensor session
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)

            #train the model
        loss, res=self.train()
        
        e = shap.DeepExplainer(([self.X1,self.X2,self.X3],self.H),[self.input1,self.input2,self.input3],session=self.sess,learning_phase_flags=[self.is_train, self.keep_prob])
        shap_values = e.shap_values([self.input1,self.input2,self.input3])

#        
        self.sess.close()
        tf.reset_default_graph()
        del self.sess
        return  loss, res,shap_values

#------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------


   

class MutiViewAutoencoder2():
    '''
      This is the implementation of the Multi-View autoencoder
    '''
    def __init__(self,data1, data2, n_hiddensh=1, activation=tf.nn.relu):

        # training datasets
        self.training_data1 = data1
        self.training_data2 = data2


        # number of features
        self.n_input1=data1.shape[1]
        self.n_input2=data2.shape[1]

        self.n_hiddensh=n_hiddensh
        self.activation = activation



#------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------
    def encode(self,X1,X2):

# =============================================================================
#         first hidden layer composed of three parts related to three sources
#          - build a layer
#          - apply the batch normalization
#          - apply a non-liner activation function
# =============================================================================

        l1= tf.layers.dense(X1, self.n_hidden1, kernel_initializer=self._init, name= 'layer1')
        l1 = tf.nn.dropout(l1, self.keep_prob)
        l1 = tf.layers.batch_normalization(l1,training=self.is_train)
        l1 = self.activation(l1)


        l2= tf.layers.dense(X2, self.n_hidden2, kernel_initializer=self._init, name= 'layer2')
        l2 = tf.nn.dropout(l2, self.keep_prob)
        l2 = tf.layers.batch_normalization(l2,training=self.is_train)
        l2 = self.activation(l2)


# =============================================================================
# fuse the parts of the first hidden alyer
# =============================================================================
        l= tf.layers.dense(tf.concat([l1,l2],1), self.n_hiddensh, kernel_initializer=self._init,
                                name= 'layer4')
        l = tf.layers.batch_normalization(l,training=self.is_train)
        l = self.activation(l)

        return l

#------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------
    def decode(self,H):

        l= tf.layers.dense(H, self.n_hidden1 +self.n_hidden2 , kernel_initializer=self._init,
                                 name= 'layer5')
        l = tf.layers.batch_normalization(l,training=self.is_train)
        l = self.activation(l)

        s1,s2 = tf.split(l,[self.n_hidden1,self.n_hidden2],1)

        l1= tf.layers.dense(s1, self.n_input1, kernel_initializer=self._init,
                                 name= 'layer6')
        l1 = tf.layers.batch_normalization(l1,training=self.is_train)
        l1 = self.activation(l1)

        l2= tf.layers.dense(s2, self.n_input2, kernel_initializer=self._init,
                                 name= 'layer7')
        l2 = tf.layers.batch_normalization(l2,training=self.is_train)
        l2 = self.activation(l2)




        return l1,l2
#------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------

    def get_weights(self):

        with tf.variable_scope("layer1", reuse=True):
            self.W1 = tf.get_variable("kernel")
        with tf.variable_scope("layer2", reuse=True):
            self.W2 = tf.get_variable("kernel")

        with tf.variable_scope("layer4", reuse=True):
            self.Wsh = tf.get_variable("kernel")
        with tf.variable_scope("layer5", reuse=True):
            self.Wsht = tf.get_variable("kernel")
        with tf.variable_scope("layer6", reuse=True):
            self.W1t = tf.get_variable("kernel")
        with tf.variable_scope("layer7", reuse=True):
            self.W2t = tf.get_variable("kernel")

#------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------

    def L1regularization(self,weights):
        return tf.reduce_sum(tf.abs(weights))

    def L2regularization(self,weights,nbunits):
        return  math.sqrt(nbunits)*tf.nn.l2_loss(weights)
#------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------
    def loss(self,X1,Y1,X2,Y2):

        self.H = self.encode(X1,X2)
        X1_,X2_=self.decode(self.H)
        self.get_weights()

        # Sparse group lasso
        sgroup_lasso = self.L2regularization(self.W1,self.n_input1* self.n_hidden1) + self.L2regularization(self.W2,self.n_input2*self.n_hidden2) 

        #Lasso
        lasso = self.L1regularization(self.W1) + self.L1regularization(self.W2) \
                       +self.L1regularization(self.Wsh)+ self.L1regularization(self.Wsht)\
                       + self.L1regularization(self.W1t) + self.L1regularization(self.W2t) 
        #Reconstruction Error
        error = tf.losses.mean_squared_error(Y1,X1_) +tf.losses.mean_squared_error(Y2,X2_)
        # Loss function
        cost= 0.5*error+ 0.5*self.lamda*(1-self.alpha)*sgroup_lasso+ 0.5*self.lamda*self.alpha*lasso

        return cost


    def corrupt(self, input_data):

        noisy_input = input_data + .2 * np.random.random_sample((input_data.shape)) - .1
        output = input_data
        return noisy_input,output
#------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------
    def Normalize(self, df, mean=None, std= None):
#        if mean is None:
#            mean = np.mean(df, axis=0)
#        if std is None:
#            std = np.std(df, axis=0,ddof=0)
#        array= np.transpose((np.transpose(df) - mean.reshape((df.shape[1],1)))/std.reshape((df.shape[1],1)))

    # Scale to [0,1]
        scaled_input_1 = np.divide((df-df.min()), (df.max()-df.min()))
#    # Scale to [-1,1]
        array = (scaled_input_1*2)-1

        return array
#------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------
    def train(self):
# =============================================================================
        # training data
            train_input1 = self.Normalize(self.input1)
            train_output1 = self.Normalize(self.output1)

            train_input2 = self.Normalize(self.input2)
            train_output2 = self.Normalize(self.output2)


# =============================================================================
# =============================================================================
            save_sess=self.sess

# =============================================================================
#           costs history :
            costs = []

            costs_inter=[]
# =============================================================================
# =============================================================================
#           for early stopping :
            best_cost=100000
            stop = False
            last_improvement=0
# =============================================================================

            n_samples = train_input1.shape[0] # size of the training set

# =============================================================================
#           train the mini_batches model using the early stopping criteria
            epoch = 0
            while epoch < self.max_epochs and stop == False:
#                train the model on the traning set by mini batches
#                   suffle then split the training set to mini-batches of size self.batch_size
                seq =list(range(n_samples))
                random.shuffle(seq)
                mini_batches = [
                    seq[k:k+self.batch_size]
                    for k in range(0,n_samples, self.batch_size)
                ]

                avg_cost = 0. # The average cost of mini_batches

                for sample in mini_batches:

                    batch_xs1 = train_input1[sample][:]
                    batch_ys1 =train_output1[sample][:]

                    batch_xs2 = train_input2[sample][:]
                    batch_ys2 = train_output2[sample][:]


                    feed_dictio={self.X1: batch_xs1,self.Y1:batch_ys1,self.X2: batch_xs2,self.Y2:batch_ys2, self.is_train:True, self.keep_prob:self.kp }
                    cost=self.sess.run([self.loss_,self.train_step], feed_dict=feed_dictio)
                    avg_cost += cost[0] *len(sample)/n_samples



#               cost history since the last best cost
                costs_inter.append(avg_cost)

                #early stopping based on the validation set/ max_steps_without_decrease of the loss value : require_improvement
                if avg_cost < best_cost:
                    save_sess= self.sess # save session
                    best_cost = avg_cost
                    costs +=costs_inter # costs history of the validatio set
                    last_improvement = 0
                    costs_inter= []
                else:
                    last_improvement +=1
                if last_improvement > self.require_improvement:
#                 #    print("No improvement found during the ( self.require_improvement) last iterations, stopping optimization.")
                    # Break out from the loop.
                     stop = True
                     self.sess=save_sess # restore session with the best cost

                epoch +=1

# =====================================End of model training ========================================
            feed_dictio={self.X1: train_input1,self.Y1:train_output1,self.X2: train_input2,self.Y2:train_output2, self.is_train:False, self.keep_prob:1 }
            costfinal, res=self.sess.run([self.loss_,self.H], feed_dict=feed_dictio)

#                normalize costs history
#            costs_val = (costs_val-min(costs_val) ) / (max(costs_val)-min(costs_val))
#            costs = (costs-min(costs) ) / (max(costs)-min(costs))
# =============================================================================
#            Display loss
           # print(epoch)
            plt.plot(costs)
            plt.ylabel('cost_test_data') ,;
            plt.xlabel('iterations')
            plt.title("Learning rate =" + str(self.learning_rate))
           # plt.show()
                   
# =============================================================================
           # print(best_cost)
           # print(costfinal)
           # res = self.sess.run(self.H, feed_dict=feed_dictio)

            return costfinal, res
#------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------
    def cross_validation(self, params):

        #retrieve parameters
        self.batch_size = params['batch_size']
        self.n_hidden1=params['units1']
        self.n_hidden2=params['units2']
        self.alpha=params['alpha']
        self.lamda=params['lamda']
        self.learning_rate=params['learning_rate']
        self.kp=params['keep_prob']
        self.require_improvement= 20
        self.max_epochs = 1000
        init = params['initializer']
        if init == 'normal':
            self._init = tf.random_normal_initializer
        if init == 'uniform':
            self._init=tf.random_uniform_initializer
        if init == 'He':
            self._init = tf.contrib.layers.variance_scaling_initializer()
        if init == 'xavier':
            self._init = tf.contrib.layers.xavier_initializer()

        opt = params['optimizer']
        if opt == 'SGD':
            self.optimizer=tf.train.GradientDescentOptimizer
        if opt == 'adam':
            self.optimizer=tf.train.AdamOptimizer
        if opt == 'nadam':
            self.optimizer=tf.contrib.opt.NadamOptimizer
        if opt == 'Momentum':
            self.optimizer=tf.train.MomentumOptimizer
        if opt == 'RMSProp':
            self.optimizer=tf.train.RMSPropOptimizer




     #   print("H.layer1:",self.n_hidden1, ", H.layer1:", self.n_hidden2)
    #    print( "k",k,"lamda", self.lamda, ", batch_size:",self.batch_size,", alpha:", self.alpha,", learning_rate:",self.learning_rate )
     #   print("initializer: ",init,'optimizer:', opt," ,kp:", self.kp)

        # add corruption to the traning set

        self.input1,self.output1 = self.corrupt(self.training_data1)
        self.input2,self.output2 = self.corrupt(self.training_data2)

        # cross-validation
       
        loss=0

            #reset tensor graph after each cross_validation run
        tf.reset_default_graph()
            
            
        self.X1=tf.placeholder("float",shape=[None,self.training_data1.shape[1]])
        self.Y1=tf.placeholder("float",shape=[None,self.training_data1.shape[1]])

        self.X2=tf.placeholder("float",shape=[None,self.training_data2.shape[1]])
        self.Y2=tf.placeholder("float",shape=[None,self.training_data2.shape[1]])
        self.keep_prob = tf.placeholder(tf.float32)


        self.is_train = tf.placeholder(tf.bool, name="is_train");

        self.loss_=self.loss(self.X1,self.Y1,self.X2,self.Y2)
        if opt == 'Momentum':
                self.train_step = self.optimizer(self.learning_rate,0.9).minimize(self.loss_)
        else:
                self.train_step = self.optimizer(self.learning_rate).minimize(self.loss_)
            # Initiate a tensor session
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)

            #train the model
        loss, res=self.train()

        e = shap.DeepExplainer(([self.X1,self.X2],self.H),[self.input1,self.input2],session=self.sess,learning_phase_flags=[self.is_train, self.keep_prob])
        shap_values = e.shap_values([self.input1,self.input2])
        

#        
        self.sess.close()
        tf.reset_default_graph()
        del self.sess
        return  loss, res,shap_values


#------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------


class MutiViewAutoencoder1():
    '''
      This is the implementation of the Multi-View autoencoder
    '''
    def __init__(self,data1, n_hiddensh=1, activation=tf.nn.relu):

        # training datasets
        self.training_data1 = data1


        # number of features
        self.n_input1=data1.shape[1]

        self.n_hiddensh=n_hiddensh
        self.activation = activation



#------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------
    def encode(self,X1):

# =============================================================================
#         first hidden layer composed of three parts related to three sources
#          - build a layer
#          - apply the batch normalization
#          - apply a non-liner activation function
# =============================================================================

        l1= tf.layers.dense(X1, self.n_hidden1, kernel_initializer=self._init, name= 'layer1')
        l1 = tf.nn.dropout(l1, self.keep_prob)
        l1 = tf.layers.batch_normalization(l1,training=self.is_train)
        l1 = self.activation(l1)


# =============================================================================
# fuse the parts of the first hidden alyer
# =============================================================================
        l= tf.layers.dense(l1, self.n_hiddensh, kernel_initializer=self._init,
                                name= 'layer4')
        l = tf.layers.batch_normalization(l,training=self.is_train)
        l = self.activation(l)

        return l
#-----------------
#------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------
    def decode(self,H):

        l= tf.layers.dense(H, self.n_hidden1  , kernel_initializer=self._init,
                                 name= 'layer5')
        l = tf.layers.batch_normalization(l,training=self.is_train)
        l = self.activation(l)


        l1= tf.layers.dense(l, self.n_input1, kernel_initializer=self._init,
                                 name= 'layer6')
        l1 = tf.layers.batch_normalization(l1,training=self.is_train)
        l1 = self.activation(l1)






        return l1
#------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------

    def get_weights(self):

        with tf.variable_scope("layer1", reuse=True):
            self.W1 = tf.get_variable("kernel")

        with tf.variable_scope("layer4", reuse=True):
            self.Wsh = tf.get_variable("kernel")
        with tf.variable_scope("layer5", reuse=True):
            self.Wsht = tf.get_variable("kernel")
        with tf.variable_scope("layer6", reuse=True):
            self.W1t = tf.get_variable("kernel")


#------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------

    def L1regularization(self,weights):
        return tf.reduce_sum(tf.abs(weights))

    def L2regularization(self,weights,nbunits):
        return  math.sqrt(nbunits)*tf.nn.l2_loss(weights)
#------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------
    def loss(self,X1,Y1):

        self.H = self.encode(X1)
        X1_=self.decode(self.H)
        self.get_weights()

        # Sparse group lasso
        sgroup_lasso = self.L2regularization(self.W1,self.n_input1* self.n_hidden1)  

        #Lasso
        lasso = self.L1regularization(self.W1)  \
                       +self.L1regularization(self.Wsh)+ self.L1regularization(self.Wsht)\
                       + self.L1regularization(self.W1t) 
        #Reconstruction Error
        error = tf.losses.mean_squared_error(Y1,X1_) 
        # Loss function
        cost= 0.5*error+ 0.5*self.lamda*(1-self.alpha)*sgroup_lasso+ 0.5*self.lamda*self.alpha*lasso

        return cost


    def corrupt(self, input_data):

        noisy_input = input_data + .2 * np.random.random_sample((input_data.shape)) - .1
        output = input_data
        return noisy_input,output
#------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------
    def Normalize(self, df, mean=None, std= None):
#        if mean is None:
#            mean = np.mean(df, axis=0)
#        if std is None:
#            std = np.std(df, axis=0,ddof=0)
#        array= np.transpose((np.transpose(df) - mean.reshape((df.shape[1],1)))/std.reshape((df.shape[1],1)))

    # Scale to [0,1]
        scaled_input_1 = np.divide((df-df.min()), (df.max()-df.min()))
#    # Scale to [-1,1]
        array = (scaled_input_1*2)-1

        return array
#------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------
    def train(self):
# =============================================================================
        # training data
            train_input1 = self.Normalize(self.input1)
            train_output1 = self.Normalize(self.output1)


# =============================================================================
# =============================================================================
            save_sess=self.sess

# =============================================================================
#           costs history :
            costs = []

            costs_inter=[]
# =============================================================================
# =============================================================================
#           for early stopping :
            best_cost=100000
            stop = False
            last_improvement=0
# =============================================================================

            n_samples = train_input1.shape[0] # size of the training set

# =============================================================================
#           train the mini_batches model using the early stopping criteria
            epoch = 0
            while epoch < self.max_epochs and stop == False:
#                train the model on the traning set by mini batches
#                   suffle then split the training set to mini-batches of size self.batch_size
                seq =list(range(n_samples))
                random.shuffle(seq)
                mini_batches = [
                    seq[k:k+self.batch_size]
                    for k in range(0,n_samples, self.batch_size)
                ]

                avg_cost = 0. # The average cost of mini_batches

                for sample in mini_batches:

                    batch_xs1 = train_input1[sample][:]
                    batch_ys1 =train_output1[sample][:]



                    feed_dictio={self.X1: batch_xs1,self.Y1:batch_ys1, self.is_train:True, self.keep_prob:self.kp }
                    cost=self.sess.run([self.loss_,self.train_step], feed_dict=feed_dictio)
                    avg_cost += cost[0] *len(sample)/n_samples



#               cost history since the last best cost
                costs_inter.append(avg_cost)

                #early stopping based on the validation set/ max_steps_without_decrease of the loss value : require_improvement
                if avg_cost < best_cost:
                    save_sess= self.sess # save session
                    best_cost = avg_cost
                    costs +=costs_inter # costs history of the validatio set
                    last_improvement = 0
                    costs_inter= []
                else:
                    last_improvement +=1
                if last_improvement > self.require_improvement:
#                  #   print("No improvement found during the ( self.require_improvement) last iterations, stopping optimization.")
                    # Break out from the loop.
                     stop = True
                     self.sess=save_sess # restore session with the best cost

                epoch +=1

# =====================================End of model training ========================================
            feed_dictio={self.X1: train_input1,self.Y1:train_output1,self.is_train:False, self.keep_prob:1 }
            costfinal, res=self.sess.run([self.loss_,self.H], feed_dict=feed_dictio)

#                normalize costs history
#            costs_val = (costs_val-min(costs_val) ) / (max(costs_val)-min(costs_val))
#            costs = (costs-min(costs) ) / (max(costs)-min(costs))
# =============================================================================
#            Display loss
          #  print(epoch)
            plt.plot(costs)
            plt.ylabel('cost_test_data') ,;
            plt.xlabel('iterations')
            plt.title("Learning rate =" + str(self.learning_rate))
            plt.show()
                   
# =============================================================================
           # print(best_cost)
            #print(costfinal)
           # res = self.sess.run(self.H, feed_dict=feed_dictio)

            return costfinal, res
#------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------
    def cross_validation(self, params):

        #retrieve parameters
        self.batch_size = params['batch_size']
        self.n_hidden1=params['units1']
        self.alpha=params['alpha']
        self.lamda=params['lamda']
        self.learning_rate=params['learning_rate']
        self.kp=params['keep_prob']
        self.require_improvement= 20
        self.max_epochs = 1000
        init= params['initializer']
        if init == 'normal':
            self._init = tf.random_normal_initializer
        if init == 'uniform':
            self._init=tf.random_uniform_initializer
        if init == 'He':
            self._init = tf.contrib.layers.variance_scaling_initializer()
        if init == 'xavier':
            self._init = tf.contrib.layers.xavier_initializer()

        opt = params['optimizer']
        if opt == 'SGD':
            self.optimizer=tf.train.GradientDescentOptimizer
        if opt == 'adam':
            self.optimizer=tf.train.AdamOptimizer
        if opt == 'nadam':
            self.optimizer=tf.contrib.opt.NadamOptimizer
        if opt == 'Momentum':
            self.optimizer=tf.train.MomentumOptimizer
        if opt == 'RMSProp':
            self.optimizer=tf.train.RMSPropOptimizer


      #  print("H.layer1:",self.n_hidden1)
       # print( "k",k,"lamda", self.lamda, ", batch_size:",self.batch_size,", alpha:", self.alpha,", learning_rate:",self.learning_rate )
        #print("initializer: ",init,'optimizer:', opt, " , kp:", self.kp)

        # add corruption to the traning set

        self.input1,self.output1 = self.corrupt(self.training_data1)

        # cross-validation
       
        loss=0

            #reset tensor graph after each cross_validation run
        tf.reset_default_graph()
            
            
        self.X1=tf.placeholder("float",shape=[None,self.training_data1.shape[1]])
        self.Y1=tf.placeholder("float",shape=[None,self.training_data1.shape[1]])
        self.keep_prob = tf.placeholder(tf.float32)



        self.is_train = tf.placeholder(tf.bool, name="is_train");

        self.loss_=self.loss(self.X1,self.Y1)
        if opt == 'Momentum':
                self.train_step = self.optimizer(self.learning_rate,0.9).minimize(self.loss_)
        else:
                self.train_step = self.optimizer(self.learning_rate).minimize(self.loss_)
            # Initiate a tensor session
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)

            #train the model
        loss, res=self.train()


        e = shap.DeepExplainer(([self.X1],self.H),[self.input1],session=self.sess,learning_phase_flags=[self.is_train, self.keep_prob])
        shap_values = e.shap_values([self.input1])
        

#        
        self.sess.close()
        tf.reset_default_graph()
        del self.sess
        return  loss, res,shap_values



if __name__=='__main__':

#    
#    selected_features=np.genfromtxt('E:/newflder/coad/featurescoad.csv', delimiter =',',skip_header =1)
##
#    inputge=np.genfromtxt('E:/newflder/coad/leukemia.csv',dtype  = np.unicode_, delimiter =',',skip_header =1)
#    namege= inputge[:,0]
#    inputge = np.transpose(inputge[:,1:inputge.shape[1]].astype(np.float))
#    
#    selected_features=np.genfromtxt('E:/newflder/selected_features_NCI_coad.csv', delimiter =',',skip_header =1)
#    inputge=np.genfromtxt('E:/newflder/data/mydatGE_coad.csv',dtype  = np.unicode_, delimiter =',',skip_header =1)
#    inputcnv=np.genfromtxt('E:/newflder/data/mydatME_coad.csv',dtype  = np.unicode_, delimiter =',',skip_header =1)
#    inputmiRNA=np.genfromtxt('E:/newflder/data/mydatMI_coad.csv',dtype  = np.unicode_, delimiter =',',skip_header =1)
####
#    inputcnv = inputcnv[:,1:inputcnv.shape[1]].astype(np.float)
#    inputge = inputge[:,1:inputge.shape[1]].astype(np.float)
#    inputmiRNA = inputmiRNA[:,1:inputmiRNA.shape[1]].astype(np.float)
##
#  
    
    with open('E:/newflder/data/pathnames.csv', 'r', encoding="ascii", errors="surrogateescape" ) as f:
          pathnames = f.readlines()
    act = tf.nn.tanh
    ii=0
  #  fname = 'E:/newflder/autre pc/trials_HPMV_maxepochs1000_max_evals50_tanh.pkl'
    fname = 'E:/newflder/autre pc/coad/tanh/COAD_trials_HPMV_maxepochs1000_max_evals50_tanh4.pkl'
  
#    fname = 'H:/Nouveau dossier (3)/autre pc/coadpins/coadp_trials_HPMV_maxepochs1000_max_evals50_tanh4.pkl'
#    namege= np.genfromtxt('E:/newflder/data/namege_coad.csv',dtype  = np.unicode_, delimiter =',',skip_header =1)
#    nameme= np.genfromtxt('E:/newflder/data/nameme_coad.csv', dtype  = np.unicode_, delimiter =',',skip_header =1)
#    namemi= np.genfromtxt('E:/newflder/data/namemi_coad.csv', dtype  = np.unicode_, delimiter =',',skip_header =1)
    input= open(fname, 'rb')

      #  print(pathnames[iii])
#    trials =pickle.load(input)

   # if fname == '/content/gdrive/My Drive/Colab Notebooks/trials_HPMV_maxepochs1000_max_evals100_p0.pkl':
  #      trials =pickle.load(input)        
   #     trials =pickle.load(input)

    # run the multiView autoencoder
    iterator =0
    for iterator in range(212) :
        pathname = pathnames[iterator+1][0:len(pathnames[iterator+1])-1].replace("\\","")
        pathname = pathname.replace("/","")
        iterators= iterator+1
        print('iteration', iterator)
        selected_feat_path = selected_features[np.where(selected_features[:,0] == iterators)[0],:]

# =============================================================================        

        print('first source ...')     
        mrna_nbr=0
        mrna_path= selected_feat_path[np.where(selected_feat_path[:,1] ==1 )[0],:]
        mrna_nbr= len(mrna_path)
        mrna_sel_data = inputge[:,mrna_path[:,2].astype(int)-1]
        nameFmrna = namege[mrna_path[:,2].astype(int)-1]


## =============================================================================        
        cnv_nbr=0
        print("second source ...")
        cnv_path= selected_feat_path[np.where(selected_feat_path[:,1] ==2 )[0],:]
        cnv_nbr= len(cnv_path)
        cnv_sel_data = inputcnv[:,cnv_path[:,2].astype(int)-1]
        nameFcnv = nameme[cnv_path[:,2].astype(int)-1]
### =============================================================================
#        print("third source ...")
        miRNA_nbr=0
        miRNA_path= selected_feat_path[np.where(selected_feat_path[:,1] == 3)[0],:]
        miRNA_nbr= len(miRNA_path)
        miRNA_sel_data = inputmiRNA[:,miRNA_path[:,2].astype(int)-1]
        nameFmi = namemi[miRNA_path[:,2].astype(int)-1]

        n_inputs1=mrna_sel_data.shape[1]
        n_inputs2=cnv_sel_data.shape[1]
        n_inputs3=miRNA_sel_data.shape[1]
        print("features size of the 1st dataset:", mrna_nbr )
        print("features size of the 2nd dataset:",cnv_nbr )
        print("features size of the 3rd dataset:",miRNA_nbr)

        n_hidden1=mrna_nbr//2+1
        n_hidden2=cnv_nbr//2+1
        n_hidden3=miRNA_nbr//2+1
        if mrna_nbr >1 or cnv_nbr >1 or miRNA_nbr>1:
          trials =pickle.load(input)
          best_loss=1000   

          best = trials.best_trial['result']['params']
        for ss in range(1):
          if mrna_nbr  >1 and miRNA_nbr<2 and cnv_nbr<2:
              sae=   MutiViewAutoencoder1(mrna_sel_data, activation = act  )

          if cnv_nbr  >1 and mrna_nbr<2 and miRNA_nbr<2:
              sae=   MutiViewAutoencoder1(cnv_sel_data, activation = act  )              
              
          if miRNA_nbr  >1 and mrna_nbr<2 and cnv_nbr<2:
              sae=   MutiViewAutoencoder1(miRNA_sel_data, activation = act  )
           
          if cnv_nbr <2 and mrna_nbr>1 and miRNA_nbr>1:


              sae=   MutiViewAutoencoder2(mrna_sel_data,miRNA_sel_data, activation = act  )


          if mrna_nbr <2 and cnv_nbr >1 and miRNA_nbr>1:

              sae=   MutiViewAutoencoder2(cnv_sel_data,miRNA_sel_data, activation = act  )



          if miRNA_nbr <2 and cnv_nbr >1 and mrna_nbr>1:

              sae=   MutiViewAutoencoder2(mrna_sel_data,cnv_sel_data, activation = act  )

          if mrna_nbr >1 and cnv_nbr >1 and miRNA_nbr>1:

              sae=   MutiViewAutoencoder(mrna_sel_data,cnv_sel_data,miRNA_sel_data, activation = act  )

            # train the HP optimization

#             get the loss of training the model on test data
          if mrna_nbr >1 or cnv_nbr >1 or miRNA_nbr>1:

            loss,h,shapfeat= sae.cross_validation(best)
            if loss< best_loss:
                   best_oss =loss
                   best_h = h
                   best_shapfeat =shapfeat
            shapfeat= best_shapfeat
        if mrna_nbr >1 or cnv_nbr >1 or miRNA_nbr>1:

          if iterator == 0   :
                  pt_path_scores = best_h
          else:
                  pt_path_scores = np.concatenate((pt_path_scores, best_h), axis =1)              
        if mrna_nbr >2 or cnv_nbr >2 or miRNA_nbr>2:

          if  mrna_nbr >1:
              if cnv_nbr  <2 and miRNA_nbr  <2:
                  fig = plt.figure()        
                  shap.summary_plot(shapfeat[0], mrna_sel_data,feature_names =nameFmrna, title= pathname +" (mRNA)", plot_type="bar")
                  fig.savefig('./inter/coad/'+pathname + 'mrna'+'.png',bbox_inches = 'tight')
                  mrnacont = np.mean(np.absolute(shapfeat[0]) , axis=0) 
                  idxmrna= mrnacont
                  resmrna = np.concatenate((nameFmrna.reshape(len(idxmrna),1),mrnacont.reshape(len(idxmrna),1)), axis =1)
                  ts= resmrna     

              else:
                  fig = plt.figure()        
                  shap.summary_plot(shapfeat[0][0], mrna_sel_data,feature_names =nameFmrna, title= pathname +" (mRNA)", plot_type="bar")
                  fig.savefig('./inter/coad/'+pathname + 'mrna'+'.png',bbox_inches = 'tight')
                  mrnacont = np.mean(np.absolute(shapfeat[0][0]) , axis=0) 
                  idxmrna= mrnacont
                  resmrna = np.concatenate((nameFmrna.reshape(len(nameFmrna),1),mrnacont.reshape(len(mrnacont),1)), axis =1)
    
                  if cnv_nbr >1:
                      fig = plt.figure()        
                      shap.summary_plot(shapfeat[0][1], cnv_sel_data,feature_names =nameFcnv, title= pathname +" (CNV)", plot_type="bar")
                      fig.savefig('./inter/coad/'+ pathname + 'cnv'+'.png',bbox_inches = 'tight')
                      cnvcont = np.mean(np.absolute(shapfeat[0][1]) , axis=0) 
                      idxcnv= cnvcont
                      rescnv = np.concatenate((nameFcnv.reshape(len(idxcnv),1),cnvcont.reshape(len(idxcnv),1)), axis =1)
                      
                      if miRNA_nbr>1:
                          fig = plt.figure()        
                          shap.summary_plot(shapfeat[0][2],miRNA_sel_data,feature_names =nameFmi, title= pathname +" (miRNA)", plot_type="bar")
                          fig.savefig('./inter/coad/'+ pathname + 'mirna'+'.png',bbox_inches = 'tight')   
                          micont = np.mean(np.absolute(shapfeat[0][2]) , axis=0) 
                          idxmi= micont
                          resmi = np.concatenate((nameFmi.reshape(len(idxmi),1),micont.reshape(len(idxmi),1)), axis =1)
                          ts= np.concatenate((resmrna,rescnv,resmi), axis = 0)
                      else:
                          ts= np.concatenate((resmrna,rescnv), axis = 0)
                          
                  else:
                      fig = plt.figure()        
                      shap.summary_plot(shapfeat[0][1],miRNA_sel_data,feature_names =nameFmi, title= pathname +" (miRNA)", plot_type="bar")
                      fig.savefig('./inter/coad/'+ pathname + 'mirna'+'.png',bbox_inches = 'tight')  
                      micont = np.mean(np.absolute(shapfeat[0][1]) , axis=0) 
                      idxmi= micont
                      resmi = np.concatenate((nameFmi.reshape(len(idxmi),1),micont.reshape(len(idxmi),1)), axis =1)
                      ts= np.concatenate((resmrna,resmi), axis = 0)                      
          else:
              if cnv_nbr >1:
                  fig = plt.figure()        

                  if miRNA_nbr>1:
                      fig = plt.figure()        
                      shap.summary_plot(shapfeat[0][0], cnv_sel_data,feature_names =nameFcnv, title= pathname + " (CNV)", plot_type="bar")
                      fig.savefig('./inter/coad/'+ pathname + 'cnv'+'.png',bbox_inches = 'tight')
                      cnvcont = np.mean(np.absolute(shapfeat[0][0] ), axis=0) 
                      idxcnv= cnvcont
                      rescnv = np.concatenate((nameFcnv.reshape(len(idxcnv),1),cnvcont.reshape(len(idxcnv),1)), axis =1)

                      fig = plt.figure()        
                      shap.summary_plot(shapfeat[0][1],miRNA_sel_data,feature_names =nameFmi, title= pathname+" (miRNA)", plot_type="bar")
                      fig.savefig('./inter/coad/'+ pathname + 'mirna'+'.png',bbox_inches = 'tight')
                      micont = np.mean(np.absolute(shapfeat[0][1]) , axis=0) 
                      idxmi= [ n for n,i in enumerate(micont) if i>0 ]
                      resmi = np.concatenate((nameFmi.reshape(len(idxmi),1),micont.reshape(len(idxmi),1)), axis =1)
                      ts= np.concatenate((rescnv,resmi), axis = 0)
                  else:
                      fig = plt.figure()        
                      shap.summary_plot(shapfeat[0], cnv_sel_data,feature_names =nameFcnv, title= pathname + " (CNV)", plot_type="bar")
                      fig.savefig('./inter/coad/'+ pathname + 'cnv'+'.png',bbox_inches = 'tight')
                      cnvcont = np.mean(np.absolute(shapfeat[0] ), axis=0) 
                      idxcnv= cnvcont
                      rescnv = np.concatenate((nameFcnv.reshape(len(idxcnv),1),cnvcont.reshape(len(idxcnv),1)), axis =1)
                      ts= np.concatenate((rescnv), axis = 0)
              else:
                  fig = plt.figure()        
                  shap.summary_plot(shapfeat[0],miRNA_sel_data,feature_names =nameFmi, title= pathname+" (miRNA)", plot_type="bar")
                  fig.savefig('./inter/coad/'+ pathname + 'mirna'+'.png',bbox_inches = 'tight')
                  micont = np.mean(np.absolute(shapfeat[0]) , axis=0) 
                  idxmi=micont
                  resmi = np.concatenate((nameFmi.reshape(len(idxmi),1),micont.reshape(len(idxmi),1)), axis =1)
                  ts= resmi     
#          del(miRNA_sel_data)
          del(mrna_sel_data)
#          del(cnv_sel_data)

          del(sae)    
          shapfeat = best_shapfeat
          
          fname = 'E:/newflder/inter/coad/SHAPValuescoadesb.pkl'
          pickle.dump(shapfeat,open( fname, "wb" ))       
          with open('E:/newflder/inter/coad/MVcoad_tanhcoadesb.csv', 'w') as csvfile:
                  writer = csv.writer(csvfile)
                  [writer.writerow(r) for r in pt_path_scores]   
#          mrna = './coadfeat/MVcoad_tanhmrnap'+str(iterators)+'.csv'
#          cnv = './coadfeat/MVcoad_tanhcnvp'+str(iterators)+'.csv'
#          mirna = './coadfeat/MVcoad_tanhmirnap'+str(iterators)+'.csv'
          

          


#          with open(mrna, 'w') as csvfile:
#                  writer = csv.writer(csvfile)
#                  writer.writerow(resmrna )
#          with open(cnv, 'w') as csvfile:
#                  writer = csv.writer(csvfile)
#                  writer.writerow( rescnv )
#          with open(mirna, 'w') as csvfile:
#                  writer = csv.writer(csvfile)
#                  writer.writerow( resmi )
          with open('E:/newflder/inter/coad/MVcoadtanhshapcoadesb.csv', 'w') as csvfile:
                  writer = csv.writer(csvfile)
                  writer.writerow( ts )
