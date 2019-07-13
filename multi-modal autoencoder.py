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



class MutiViewAutoencoder():
    '''
      This is the implementation of the Multi-modal autoencoder
    '''
    def __init__(self,data1, data2, data3,testdata1=None,testdata2=None,testdata3= None, n_hiddensh=1, activation=tf.nn.relu):

        # training datasets
        self.training_data1 = data1
        self.training_data2 = data2
        self.training_data3 = data3
        #test datasets
        self.test_data1 = testdata1
        self.test_data2 = testdata2
        self.test_data3 = testdata3

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



        l3= tf.layers.dense(X3, self.n_hidden3, kernel_initializer=self._init, name= 'layer3')
        l3 = tf.nn.dropout(l3, self.keep_prob)
        l3 = tf.layers.batch_normalization(l3,training=self.is_train)
        l3 = self.activation(l3)

# =============================================================================
# fuse the parts of the first hidden layer
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

        l= tf.layers.dense(H, self.n_hidden1 +self.n_hidden2 +self.n_hidden3, kernel_initializer=self._init,name= 'layer5')
        l = tf.layers.batch_normalization(l,training=self.is_train)
        l = self.activation(l)

        s1,s2,s3 = tf.split(l,[self.n_hidden1,self.n_hidden2,self.n_hidden3],1)

        l1= tf.layers.dense(s1, self.n_input1, kernel_initializer=self._init, name= 'layer6')
        l1 = tf.layers.batch_normalization(l1,training=self.is_train)
        l1 = self.activation(l1)

        l2= tf.layers.dense(s2, self.n_input2, kernel_initializer=self._init, name= 'layer7')
        l2 = tf.layers.batch_normalization(l2,training=self.is_train)
        l2 = self.activation(l2)

        l3= tf.layers.dense(s3, self.n_input3, kernel_initializer=self._init, name= 'layer8')
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
    def train(self,train_index,val_index):
# =============================================================================
#        training data
            train_input1 = self.Normalize(self.input1[train_index,:])
            train_output1 = self.Normalize(self.output1[train_index,:])

            train_input2 = self.Normalize(self.input2[train_index,:])
            train_output2 = self.Normalize(self.output2[train_index,:])

            train_input3 = self.Normalize(self.input3[train_index,:])
            train_output3 = self.Normalize(self.output3[train_index,:])
# =============================================================================
# =============================================================================
#         validation data

            val_input1 = self.Normalize(self.input1[val_index,:], np.mean(self.input1[train_index,:], axis=0), np.std(self.input1[train_index,:], axis=0))
            val_output1 = self.Normalize(self.output1[val_index,:], np.mean(self.output1[train_index,:], axis=0), np.std(self.output1[train_index,:], axis=0))

            val_input2 = self.Normalize(self.input2[val_index,:], np.mean(self.input2[train_index,:], axis=0), np.std(self.input2[train_index,:], axis=0))
            val_output2 = self.Normalize(self.output2[val_index,:], np.mean(self.output2[train_index,:], axis=0), np.std(self.output2[train_index,:], axis=0))

            val_input3 = self.Normalize(self.input3[val_index,:], np.mean(self.input3[train_index,:], axis=0), np.std(self.input3[train_index,:], axis=0))
            val_output3 = self.Normalize(self.output3[val_index,:], np.mean(self.output3[train_index,:], axis=0), np.std(self.output3[train_index,:], axis=0))
# =============================================================================
            save_sess=self.sess

# =============================================================================
#           costs history :
            costs = []
            costs_val = []
            costs_val_inter = []
            costs_inter=[]
# =============================================================================
# =============================================================================
#           for early stopping :
            best_cost=0
            best_val_cost = 100000
            stop = False
            last_improvement=0
# =============================================================================

            n_samples = train_input1.shape[0] # size of the training set
            vn_samples = val_input1.shape[0]  # size of the validation set

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

#                train the model on the validation set by mini batches
#                   Split the validation set to mini-batches of size self.batch_size
                seq =list(range(vn_samples))
                mini_batches = [
                    seq[k:k+self.batch_size]
                    for k in range(0,vn_samples, self.batch_size)
                ]
                avg_cost_val = 0.

                for sample in mini_batches:

                    batch_xs1 = val_input1[sample][:]
                    batch_ys1 =val_output1[sample][:]

                    batch_xs2 = val_input2[sample][:]
                    batch_ys2 = val_output2[sample][:]

                    batch_xs3 = val_input3[sample][:]
                    batch_ys3 = val_output3[sample][:]

                    feed_dictio={self.X1: batch_xs1,self.Y1:batch_ys1,self.X2: batch_xs2,self.Y2:batch_ys2,self.X3: batch_xs3,self.Y3:batch_ys3, self.is_train:False, self.keep_prob:1 }
                    cost_val= self.sess.run(self.loss_, feed_dict=feed_dictio)
                    avg_cost_val += cost_val*len(sample)/vn_samples

#               cost history since the last best cost
                costs_val_inter.append(avg_cost_val)
                costs_inter.append(avg_cost)

#               early stopping based on the validation set/ max_steps_without_decrease of the loss value : require_improvement
                if avg_cost_val < best_val_cost:
                    save_sess= self.sess # save session
                    best_val_cost = avg_cost_val
                    costs_val +=costs_val_inter # costs history of the validatio set
                    costs+=costs_inter # costs history of the training set
                    last_improvement = 0
                    costs_val_inter= []
                    costs_inter=[]
                    best_cost= avg_cost
                else:
                    last_improvement +=1
                if last_improvement > self.require_improvement:
#               print("No improvement found during the ( self.require_improvement) last iterations, stopping optimization.")
#               Break out from the loop.
                     stop = True
                     self.sess=save_sess # restore session with the best cost

                epoch +=1

# =====================================End of model training ========================================


#            normalize costs history
#            costs_val = (costs_val-min(costs_val) ) / (max(costs_val)-min(costs_val))
#            costs = (costs-min(costs) ) / (max(costs)-min(costs))
# =============================================================================
#            Display loss

# =============================================================================

            self.histcosts= costs
            self.histvalcosts=costs_val
            return best_cost,best_val_cost
#------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------
    def cross_validation(self, params):

#       retrieve parameters
        self.batch_size = params['batch_size']
        self.n_hidden1=params['units1']
        self.n_hidden2=params['units2']
        self.n_hidden3=params['units3']
        self.alpha=params['alpha']
        self.lamda=params['lamda']
        self.kp=params['keep_prob']
        self.learning_rate=params['learning_rate']
        k=5
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

#       add corruption to the traning set

        self.input1,self.output1 = self.corrupt(self.training_data1)
        self.input2,self.output2 = self.corrupt(self.training_data2)
        self.input3,self.output3 = self.corrupt(self.training_data3)

#       cross-validation
        data = np.concatenate([self.input1,self.input2,self.input3], axis=1)
        kf = KFold(n_splits=k)
        kf.get_n_splits(data) # returns the number of splitting iterations in the cross-validator

        loss_cv=0
        val_loss_cv=0

        for train_index, val_index in kf.split(data):
#       reset tensor graph after each cross_validation run
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
            loss_cv,val_loss_cv=self.train(train_index,val_index)

            loss_cv += loss_cv
            val_loss_cv += val_loss_cv

        loss_cv= loss_cv/k
        val_loss_cv=val_loss_cv/k

        hist_costs= self.histcosts
        hist_val_costs= self.histvalcosts

        self.sess.close()
        tf.reset_default_graph()
        del self.sess

        return  {'loss': val_loss_cv, 'status': STATUS_OK,'params': params,'loss_train':loss_cv,'history_loss': hist_costs,'history_val_loss': hist_val_costs}

#------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------


    def train_test(self,params):

        batch_size = params['batch_size']
        self.n_hidden1=params['units1']
        self.n_hidden2=params['units2']
        self.n_hidden3=params['units3']
        self.alpha=params['alpha']
        self.lamda=params['lamda']
        self.kp=params['keep_prob']
        self.learning_rate=params['learning_rate']
        
        learning_rate=params['learning_rate']
        k=5
        
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



        # add corruption to the test set
        input1,output1 = self.corrupt(self.test_data1)
        input2,output2 = self.corrupt(self.test_data2)
        input3,output3 = self.corrupt(self.test_data3)
        # normalize test data
        test_input1 = self.Normalize(input1)
        test_output1 = self.Normalize(output1)

        test_input2 = self.Normalize(input2)
        test_output2 = self.Normalize(output2)

        test_input3 = self.Normalize(input3)
        test_output3 = self.Normalize(output3)
        
        # tensor variables
        X1=tf.placeholder("float",shape=[None,self.test_data1.shape[1]])
        Y1=tf.placeholder("float",shape=[None,self.test_data1.shape[1]])

        X2=tf.placeholder("float",shape=[None,self.test_data2.shape[1]])
        Y2=tf.placeholder("float",shape=[None,self.test_data2.shape[1]])

        X3=tf.placeholder("float",shape=[None,self.test_data3.shape[1]])
        Y3=tf.placeholder("float",shape=[None,self.test_data3.shape[1]])
        self.is_train = tf.placeholder(tf.bool, name="is_train");
        self.keep_prob = tf.placeholder(tf.float32)

        #train the model
        loss_=self.loss(X1,Y1,X2,Y2,X3,Y3)
        if opt == 'Momentum':
            train_step = self.optimizer(learning_rate,0.9).minimize(loss_)
        else:
            train_step = self.optimizer(learning_rate).minimize(loss_)

        # Initiate a tensor session
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)
        save_sess=self.sess

        costs = []
        costs_inter = []
        data = np.concatenate([input1,input2,input3], axis=1)
        kf = KFold(n_splits=k)
        kf.get_n_splits(data) # returns the number of splitting iterations in the cross-validator

        epoch = 0
        best_cost=100000
        stop = False
        n_samples = test_input1.shape[0]
        last_improvement=0

        while epoch < self.max_epochs and stop == False:
#                train the model on the test set by mini batches
#                   shuffle then split the test set into mini-batches of size self.batch_size
                seq =list(range(n_samples))
                random.shuffle(seq)
                mini_batches = [
                    seq[k:k+batch_size]
                    for k in range(0,n_samples,batch_size)
                ]
                avg_cost = 0.

                # Loop over all batches
                for sample in mini_batches:

                    batch_xs1 = test_input1[sample][:]
                    batch_ys1 =test_output1[sample][:]

                    batch_xs2 = test_input2[sample][:]
                    batch_ys2 = test_output2[sample][:]

                    batch_xs3 = test_input3[sample][:]
                    batch_ys3 = test_output3[sample][:]

                    feed_dictio={X1: batch_xs1,Y1:batch_ys1,X2: batch_xs2,Y2:batch_ys2,X3: batch_xs3,Y3:batch_ys3, self.is_train:True, self.keep_prob:self.kp }
                    cost=self.sess.run([loss_,train_step], feed_dict=feed_dictio)
                    avg_cost += cost[0]* len(sample)/ n_samples

                costs_inter.append(avg_cost)

                #early stopping based on the validation data/ max_steps_without_decrease of the loss value : require_improvement
                if avg_cost < best_cost:
                    save_sess= self.sess
                    best_cost = avg_cost
                    costs+=costs_inter
                    last_improvement = 0
                    costs_inter= []
                else:
                    last_improvement +=1
                if last_improvement > self.require_improvement:
#                     print("No improvement found in a while, stopping optimization.")
                    # Break out from the loop.
                     stop = True
                     self.sess=save_sess
                epoch +=1

        self.sess.close()
        tf.reset_default_graph()
        del self.sess
        
        return  best_cost






class MutiViewAutoencoder2():
    '''
      This is the implementation of the Multi-View autoencoder
    '''
    def __init__(self,data1, data2,testdata1=None,testdata2=None, n_hiddensh=1, activation=tf.nn.relu):

        # training datasets
        self.training_data1 = data1
        self.training_data2 = data2
        #test datasets
        self.test_data1 = testdata1
        self.test_data2 = testdata2

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
        lasso = self.L1regularization(self.W1) + self.L1regularization(self.W2)  \
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
    def train(self,train_index,val_index):
# =============================================================================
        # training data
            train_input1 = self.Normalize(self.input1[train_index,:])
            train_output1 = self.Normalize(self.output1[train_index,:])

            train_input2 = self.Normalize(self.input2[train_index,:])
            train_output2 = self.Normalize(self.output2[train_index,:])

# =============================================================================
# =============================================================================
#            validation data

            val_input1 = self.Normalize(self.input1[val_index,:], np.mean(self.input1[train_index,:], axis=0), np.std(self.input1[train_index,:], axis=0))
            val_output1 = self.Normalize(self.output1[val_index,:], np.mean(self.output1[train_index,:], axis=0), np.std(self.output1[train_index,:], axis=0))

            val_input2 = self.Normalize(self.input2[val_index,:], np.mean(self.input2[train_index,:], axis=0), np.std(self.input2[train_index,:], axis=0))
            val_output2 = self.Normalize(self.output2[val_index,:], np.mean(self.output2[train_index,:], axis=0), np.std(self.output2[train_index,:], axis=0))
# =============================================================================
            save_sess=self.sess

# =============================================================================
#           costs history :
            costs = []
            costs_val = []
            costs_val_inter = []
            costs_inter=[]
# =============================================================================
# =============================================================================
#           for early stopping :
            best_cost=0
            best_val_cost = 100000
            stop = False
            last_improvement=0
# =============================================================================

            n_samples = train_input1.shape[0] # size of the training set
            vn_samples = val_input1.shape[0]  # size of the validation set

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

#                train the model on the validation set by mini batches
#                   Split the validation set to mini-batches of size self.batch_size
                seq =list(range(vn_samples))
                mini_batches = [
                    seq[k:k+self.batch_size]
                    for k in range(0,vn_samples, self.batch_size)
                ]
                avg_cost_val = 0.

                for sample in mini_batches:

                    batch_xs1 = val_input1[sample][:]
                    batch_ys1 =val_output1[sample][:]

                    batch_xs2 = val_input2[sample][:]
                    batch_ys2 = val_output2[sample][:]



                    feed_dictio={self.X1: batch_xs1,self.Y1:batch_ys1,self.X2: batch_xs2,self.Y2:batch_ys2, self.is_train:False, self.keep_prob:1 }
                    cost_val= self.sess.run(self.loss_, feed_dict=feed_dictio)
                    avg_cost_val += cost_val*len(sample)/vn_samples

#               cost history since the last best cost
                costs_val_inter.append(avg_cost_val)
                costs_inter.append(avg_cost)

                #early stopping based on the validation set/ max_steps_without_decrease of the loss value : require_improvement
                if avg_cost_val < best_val_cost:
                    save_sess= self.sess # save session
                    best_val_cost = avg_cost_val
                    costs_val +=costs_val_inter # costs history of the validatio set
                    costs+=costs_inter # costs history of the training set
                    last_improvement = 0
                    costs_val_inter= []
                    costs_inter=[]
                    best_cost= avg_cost
                else:
                    last_improvement +=1
                if last_improvement > self.require_improvement:
#                     print("No improvement found during the ( self.require_improvement) last iterations, stopping optimization.")
                    # Break out from the loop.
                     stop = True
                     self.sess=save_sess # restore session with the best cost

                epoch +=1

# =====================================End of model training ========================================


#                normalize costs history
#            costs_val = (costs_val-min(costs_val) ) / (max(costs_val)-min(costs_val))
#            costs = (costs-min(costs) ) / (max(costs)-min(costs))
# =============================================================================
#            Display loss

# =============================================================================

            self.histcosts= costs
            self.histvalcosts=costs_val
            return best_cost,best_val_cost
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
        k=5
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




        # add corruption to the traning set

        self.input1,self.output1 = self.corrupt(self.training_data1)
        self.input2,self.output2 = self.corrupt(self.training_data2)

        # cross-validation
        data = np.concatenate([self.input1,self.input2], axis=1)
        kf = KFold(n_splits=k)
        kf.get_n_splits(data) # returns the number of splitting iterations in the cross-validator

        loss_cv=0
        val_loss_cv=0

        for train_index, val_index in kf.split(data):
            #reset tensor graph after each cross_validation run
            tf.reset_default_graph()
            
            
            self.X1=tf.placeholder("float",shape=[None,self.training_data1.shape[1]])
            self.Y1=tf.placeholder("float",shape=[None,self.training_data1.shape[1]])

            self.X2=tf.placeholder("float",shape=[None,self.training_data2.shape[1]])
            self.Y2=tf.placeholder("float",shape=[None,self.training_data2.shape[1]])

            self.is_train = tf.placeholder(tf.bool, name="is_train");
            self.keep_prob = tf.placeholder(tf.float32)

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
            loss_cv,val_loss_cv=self.train(train_index,val_index)

            loss_cv += loss_cv
            val_loss_cv += val_loss_cv

        loss_cv= loss_cv/k
        val_loss_cv=val_loss_cv/k

        hist_costs= self.histcosts
        hist_val_costs= self.histvalcosts

        self.sess.close()
        tf.reset_default_graph()
        del self.sess

        return  {'loss': val_loss_cv, 'status': STATUS_OK,'params': params,'loss_train':loss_cv,'history_loss': hist_costs,'history_val_loss': hist_val_costs}

#------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------


    def train_test(self,params):

        batch_size = params['batch_size']
        self.n_hidden1=params['units1']
        self.n_hidden2=params['units2']
        self.alpha=params['alpha']
        self.lamda=params['lamda']
        self.learning_rate=params['learning_rate']
        self.kp=params['keep_prob']
        
        learning_rate=params['learning_rate']
        k=5
        
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



        # add corruption to the test set
        input1,output1 = self.corrupt(self.test_data1)
        input2,output2 = self.corrupt(self.test_data2)
        # normalize test data
        test_input1 = self.Normalize(input1)
        test_output1 = self.Normalize(output1)

        test_input2 = self.Normalize(input2)
        test_output2 = self.Normalize(output2)


        
        # tensor variables
        X1=tf.placeholder("float",shape=[None,self.test_data1.shape[1]])
        Y1=tf.placeholder("float",shape=[None,self.test_data1.shape[1]])

        X2=tf.placeholder("float",shape=[None,self.test_data2.shape[1]])
        Y2=tf.placeholder("float",shape=[None,self.test_data2.shape[1]])


        self.is_train = tf.placeholder(tf.bool, name="is_train");
        self.keep_prob = tf.placeholder(tf.float32)

        #train the model
        loss_=self.loss(X1,Y1,X2,Y2)
        if opt == 'Momentum':
            train_step = self.optimizer(learning_rate,0.9).minimize(loss_)
        else:
            train_step = self.optimizer(learning_rate).minimize(loss_)

        # Initiate a tensor session
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)
        save_sess=self.sess
        costs = []
        costs_inter = []
        data = np.concatenate([input1,input2], axis=1)
        kf = KFold(n_splits=k)
        kf.get_n_splits(data) # returns the number of splitting iterations in the cross-validator

        epoch = 0
        best_cost=100000
        stop = False
        n_samples = test_input1.shape[0]
        last_improvement=0

        while epoch < self.max_epochs and stop == False:
#                train the model on the test set by mini batches
#                   suffle then split the test set to mini-batches of size self.batch_size
                seq =list(range(n_samples))
                random.shuffle(seq)
                mini_batches = [
                    seq[k:k+batch_size]
                    for k in range(0,n_samples,batch_size)
                ]
                avg_cost = 0.

                # Loop over all batches
                for sample in mini_batches:

                    batch_xs1 = test_input1[sample][:]
                    batch_ys1 =test_output1[sample][:]

                    batch_xs2 = test_input2[sample][:]
                    batch_ys2 = test_output2[sample][:]



                    feed_dictio={X1: batch_xs1,Y1:batch_ys1,X2: batch_xs2,Y2:batch_ys2,self.is_train:True, self.keep_prob:self.kp }
                    cost=self.sess.run([loss_,train_step], feed_dict=feed_dictio)
                    avg_cost += cost[0]* len(sample)/ n_samples

                costs_inter.append(avg_cost)

                #early stopping based on the validation data/ max_steps_without_decrease of the loss value : require_improvement
                if avg_cost < best_cost:
                    save_sess= self.sess
                    best_cost = avg_cost
                    costs+=costs_inter
                    last_improvement = 0
                    costs_inter= []
                else:
                    last_improvement +=1
                if last_improvement > self.require_improvement:
#                     print("No improvement found in a while, stopping optimization.")
                    # Break out from the loop.
                     stop = True
                     self.sess=save_sess
                epoch +=1

        self.sess.close()
        tf.reset_default_graph()
        del self.sess
        
        return  best_cost







class MutiViewAutoencoder1():
    '''
      This is the implementation of the Multi-View autoencoder
    '''
    def __init__(self,data1,testdata1=None, n_hiddensh=1, activation=tf.nn.relu):

        # training datasets
        self.training_data1 = data1
        #test datasets
        self.test_data1 = testdata1

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
#------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------
    def decode(self,H):

        l= tf.layers.dense(H, self.n_hidden1 , kernel_initializer=self._init,
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
    def train(self,train_index,val_index):
# =============================================================================
        # training data
            train_input1 = self.Normalize(self.input1[train_index,:])
            train_output1 = self.Normalize(self.output1[train_index,:])



# =============================================================================
# =============================================================================
#            validation data

            val_input1 = self.Normalize(self.input1[val_index,:], np.mean(self.input1[train_index,:], axis=0), np.std(self.input1[train_index,:], axis=0))
            val_output1 = self.Normalize(self.output1[val_index,:], np.mean(self.output1[train_index,:], axis=0), np.std(self.output1[train_index,:], axis=0))

# =============================================================================
            save_sess=self.sess

# =============================================================================
#           costs history :
            costs = []
            costs_val = []
            costs_val_inter = []
            costs_inter=[]
# =============================================================================
# =============================================================================
#           for early stopping :
            best_cost=0
            best_val_cost = 100000
            stop = False
            last_improvement=0
# =============================================================================

            n_samples = train_input1.shape[0] # size of the training set
            vn_samples = val_input1.shape[0]  # size of the validation set

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

#                train the model on the validation set by mini batches
#                   Split the validation set to mini-batches of size self.batch_size
                seq =list(range(vn_samples))
                mini_batches = [
                    seq[k:k+self.batch_size]
                    for k in range(0,vn_samples, self.batch_size)
                ]
                avg_cost_val = 0.

                for sample in mini_batches:

                    batch_xs1 = val_input1[sample][:]
                    batch_ys1 =val_output1[sample][:]




                    feed_dictio={self.X1: batch_xs1,self.Y1:batch_ys1, self.is_train:False , self.keep_prob:1}
                    cost_val= self.sess.run(self.loss_, feed_dict=feed_dictio)
                    avg_cost_val += cost_val*len(sample)/vn_samples

#               cost history since the last best cost
                costs_val_inter.append(avg_cost_val)
                costs_inter.append(avg_cost)

                #early stopping based on the validation set/ max_steps_without_decrease of the loss value : require_improvement
                if avg_cost_val < best_val_cost:
                    save_sess= self.sess # save session
                    best_val_cost = avg_cost_val
                    costs_val +=costs_val_inter # costs history of the validatio set
                    costs+=costs_inter # costs history of the training set
                    last_improvement = 0
                    costs_val_inter= []
                    costs_inter=[]
                    best_cost= avg_cost
                else:
                    last_improvement +=1
                if last_improvement > self.require_improvement:
#                     print("No improvement found during the ( self.require_improvement) last iterations, stopping optimization.")
                    # Break out from the loop.
                     stop = True
                     self.sess=save_sess # restore session with the best cost

                epoch +=1

# =====================================End of model training ========================================


#                normalize costs history
#            costs_val = (costs_val-min(costs_val) ) / (max(costs_val)-min(costs_val))
#            costs = (costs-min(costs) ) / (max(costs)-min(costs))
# =============================================================================
#            Display loss

# =============================================================================

            self.histcosts= costs
            self.histvalcosts=costs_val
            return best_cost,best_val_cost
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
        k=5
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




        # add corruption to the traning set

        self.input1,self.output1 = self.corrupt(self.training_data1)

        # cross-validation
        data = self.input1
        kf = KFold(n_splits=k)
        kf.get_n_splits(data) # returns the number of splitting iterations in the cross-validator

        loss_cv=0
        val_loss_cv=0

        for train_index, val_index in kf.split(data):
            #reset tensor graph after each cross_validation run
            tf.reset_default_graph()
            
            
            self.X1=tf.placeholder("float",shape=[None,self.training_data1.shape[1]])
            self.Y1=tf.placeholder("float",shape=[None,self.training_data1.shape[1]])



            self.is_train = tf.placeholder(tf.bool, name="is_train");
            self.keep_prob = tf.placeholder(tf.float32)

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
            loss_cv,val_loss_cv=self.train(train_index,val_index)

            loss_cv += loss_cv
            val_loss_cv += val_loss_cv

        loss_cv= loss_cv/k
        val_loss_cv=val_loss_cv/k

        hist_costs= self.histcosts
        hist_val_costs= self.histvalcosts

        self.sess.close()
        tf.reset_default_graph()
        del self.sess

        return  {'loss': val_loss_cv, 'status': STATUS_OK,'params': params,'loss_train':loss_cv,'history_loss': hist_costs,'history_val_loss': hist_val_costs}

#------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------


    def train_test(self,params):

        batch_size = params['batch_size']
        self.n_hidden1=params['units1']
        self.alpha=params['alpha']
        self.lamda=params['lamda']
        self.learning_rate=params['learning_rate']
        self.kp=params['keep_prob']
        
        learning_rate=params['learning_rate']
        k=5
        
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



        # add corruption to the test set
        input1,output1 = self.corrupt(self.test_data1)
        # normalize test data
        test_input1 = self.Normalize(input1)
        test_output1 = self.Normalize(output1)



        
        # tensor variables
        X1=tf.placeholder("float",shape=[None,self.test_data1.shape[1]])
        Y1=tf.placeholder("float",shape=[None,self.test_data1.shape[1]])
        self.keep_prob = tf.placeholder(tf.float32)
        self.is_train = tf.placeholder(tf.bool, name="is_train");

        #train the model
        loss_=self.loss(X1,Y1)
        if opt == 'Momentum':
            train_step = self.optimizer(learning_rate,0.9).minimize(loss_)
        else:
            train_step = self.optimizer(learning_rate).minimize(loss_)

        # Initiate a tensor session
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)
        save_sess=self.sess
        costs = []
        costs_inter = []
        data = input1
        kf = KFold(n_splits=k)
        kf.get_n_splits(data) # returns the number of splitting iterations in the cross-validator

        epoch = 0
        best_cost=100000
        stop = False
        n_samples = test_input1.shape[0]
        last_improvement=0

        while epoch < self.max_epochs and stop == False:
#                train the model on the test set by mini batches
#                   shuffle then split the test set to mini-batches of size self.batch_size
                seq =list(range(n_samples))
                random.shuffle(seq)
                mini_batches = [
                    seq[k:k+batch_size]
                    for k in range(0,n_samples,batch_size)
                ]
                avg_cost = 0.

                # Loop over all batches
                for sample in mini_batches:

                    batch_xs1 = test_input1[sample][:]
                    batch_ys1 =test_output1[sample][:]




                    feed_dictio={X1: batch_xs1,Y1:batch_ys1,self.is_train:True, self.keep_prob:self.kp }
                    cost=self.sess.run([loss_,train_step], feed_dict=feed_dictio)
                    avg_cost += cost[0]* len(sample)/ n_samples

                costs_inter.append(avg_cost)

                #early stopping based on the validation data/ max_steps_without_decrease of the loss value : require_improvement
                if avg_cost < best_cost:
                    save_sess= self.sess
                    best_cost = avg_cost
                    costs+=costs_inter
                    last_improvement = 0
                    costs_inter= []
                else:
                    last_improvement +=1
                if last_improvement > self.require_improvement:
#                     print("No improvement found in a while, stopping optimization.")
                    # Break out from the loop.
                     stop = True
                     self.sess=save_sess
                epoch +=1

                # Display loss per epoch step

                    # Display logs per epoch



        self.sess.close()
        tf.reset_default_graph()
        del self.sess
        
        return  best_cost

if __name__=='__main__':

# datasets: RNA,miRNA,DNA_Methy/ NCI pathways
#
    selected_features=np.genfromtxt('./selected_features_NCI_coad9918.csv', delimiter =',',skip_header =1)

    inputge=np.genfromtxt('./COAD/COADinputge.csv',dtype  = np.unicode_, delimiter =',',skip_header =1)
    inputcnv=np.genfromtxt('./COAD/COADinputcnv.csv',dtype  = np.unicode_, delimiter =',',skip_header =1)
    inputmiRNA=np.genfromtxt('./COAD/COADinputmiRNA513.csv',dtype  = np.unicode_, delimiter =',',skip_header =1)    

    inputcnv = inputcnv[:,2:inputcnv.shape[1]].astype(np.float)
    inputge = inputge[:,2:inputge.shape[1]].astype(np.float)
    inputmiRNA = inputmiRNA[:,2:inputmiRNA.shape[1]].astype(np.float)

    act = tf.nn.tanh
    ii=0
    # run the multiView autoencoder for the 212 NCI pathways
    for iterator in range(212) :

        print('iteration', iterator)
        selected_feat_path = selected_features[np.where(selected_features[:,0] == iterator+1)[0],:]
# =============================================================================        

        print('first source ...')        
        mrna_nbr=0
        mrna_path = selected_feat_path[np.where(selected_feat_path[:,1] == 1)[0],:]
        mrna_nbr= len(mrna_path)
        mrna_sel_data = inputge[:,mrna_path[:,2].astype(int)-1]
# =============================================================================        
        cnv_nbr=0
        print("second source ...")
        cnv_path= selected_feat_path[np.where(selected_feat_path[:,1] ==2 )[0],:]
        cnv_nbr= len(cnv_path)
        cnv_sel_data = inputcnv[:,cnv_path[:,2].astype(int)-1]
## =============================================================================
        print("third source ...")
        miRNA_nbr=0
        miRNA_path= selected_feat_path[np.where(selected_feat_path[:,1] == 3)[0],:]
        miRNA_nbr= len(miRNA_path)
        miRNA_sel_data = inputmiRNA[:,miRNA_path[:,2].astype(int)-1]

        n_inputs1=mrna_sel_data.shape[1]
        n_inputs2=cnv_sel_data.shape[1]
        n_inputs3=miRNA_sel_data.shape[1]
        print("features size of the 1st dataset:", mrna_nbr )
        print("features size of the 2nd dataset:",cnv_nbr )
        print("features size of the 3rd dataset:",miRNA_nbr)

        n_hidden1=mrna_nbr//2+1
        n_hidden2=cnv_nbr//2+1
        n_hidden3=miRNA_nbr//2+1
        if mrna_nbr >1 and cnv_nbr<2 and miRNA_nbr<2:
            #Split dataset into training and test data 80%/20%
            X_train1, X_test1 = model_selection.train_test_split(mrna_sel_data, test_size=0.2, random_state=1)

            sae=   MutiViewAutoencoder1(X_train1,X_test1, activation = act  )
            trials = Trials()
            #define the space of hyperparameters
            space = {
                     'units1': hp.choice('units1', range(1,n_hidden1)),
                     'batch_size': hp.choice('batch_size',[16,8,4] ) ,
                     'alpha' :hp.choice('alpha',[0, hp.uniform('alpha2',0,1)]),
                     'learning_rate' : hp.loguniform('learning_rate',-5,-1),
                     'lamda':hp.choice('lamda',[0,hp.loguniform('lamda2',-8,-1)]),
                     'optimizer':hp.choice('optimizer',["adam","nadam","SGD","Momentum","RMSProp"]),
                     'initializer':hp.choice('initializer',["normal","uniform","xavier","He"]),
                     'keep_prob':hp.choice('keep_prob',[1, hp.uniform('kp',0.5,1)])

                    }        
        if cnv_nbr >1 and mrna_nbr<2 and miRNA_nbr<2:
            #Split dataset to training and test data 80%/20%
            X_train1, X_test1 = model_selection.train_test_split(cnv_sel_data, test_size=0.2, random_state=1)

            sae=   MutiViewAutoencoder1(X_train1,X_test1, activation = act  )
            trials = Trials()
            #define the space of hyperparameters
            space = {
                     'units1': hp.choice('units1', range(1,n_hidden2)),
                     'batch_size': hp.choice('batch_size',[16,8,4] ) ,
                     'alpha' :hp.choice('alpha',[0, hp.uniform('alpha2',0,1)]),
                     'learning_rate' : hp.loguniform('learning_rate',-5,-1),
                     'lamda':hp.choice('lamda',[0,hp.loguniform('lamda2',-8,-1)]),
                     'optimizer':hp.choice('optimizer',["adam","nadam","SGD","Momentum","RMSProp"]),
                     'initializer':hp.choice('initializer',["normal","uniform","xavier","He"]),
                     'keep_prob':hp.choice('keep_prob',[1, hp.uniform('kp',0.5,1)])

                    }        
        if miRNA_nbr >1 and mrna_nbr<2 and cnv_nbr<2:
            #Split dataset to training and test data 80%/20%
            X_train1, X_test1 = model_selection.train_test_split(miRNA_sel_data, test_size=0.2, random_state=1)

            sae=   MutiViewAutoencoder1(X_train1,X_test1, activation = act  )
            trials = Trials()
            #define the space of hyperparameters
            space = {
                     'units1': hp.choice('units1', range(1,n_hidden3)),
                     'batch_size': hp.choice('batch_size',[16,8,4] ) ,
                     'alpha' :hp.choice('alpha',[0, hp.uniform('alpha2',0,1)]),
                     'learning_rate' : hp.loguniform('learning_rate',-5,-1),
                     'lamda':hp.choice('lamda',[0,hp.loguniform('lamda2',-8,-1)]),
                     'optimizer':hp.choice('optimizer',["adam","nadam","SGD","Momentum","RMSProp"]),
                     'initializer':hp.choice('initializer',["normal","uniform","xavier","He"]),
                     'keep_prob':hp.choice('keep_prob',[1, hp.uniform('kp',0.5,1)])

                    }            
        if cnv_nbr <2 and mrna_nbr>1 and miRNA_nbr>1:
            #Split dataset to training and test data 80%/20%
            X_train1, X_test1 = model_selection.train_test_split(mrna_sel_data, test_size=0.2, random_state=1)
            X_train3, X_test3 = model_selection.train_test_split(miRNA_sel_data, test_size=0.2, random_state=1)

            sae=   MutiViewAutoencoder2(X_train1,X_train3,X_test1,X_test3, activation = act  )
            trials = Trials()
            #define the space of hyperparameters
            space = {
                     'units1': hp.choice('units1', range(1,n_hidden1)),
                     'units2': hp.choice('units2', range(1,n_hidden3)),
                     'batch_size': hp.choice('batch_size',[16,8,4] ) ,
                     'alpha' :hp.choice('alpha',[0, hp.uniform('alpha2',0,1)]),
                     'learning_rate' : hp.loguniform('learning_rate',-5,-1),
                     'lamda':hp.choice('lamda',[0,hp.loguniform('lamda2',-8,-1)]),
                     'optimizer':hp.choice('optimizer',["adam","nadam","SGD","Momentum","RMSProp"]),
                     'initializer':hp.choice('initializer',["normal","uniform","xavier","He"]),
                     'keep_prob':hp.choice('keep_prob',[1, hp.uniform('kp',0.5,1)])

                    }        
        if mrna_nbr <2 and cnv_nbr >1 and miRNA_nbr>1:
            #Split dataset to training and test data 80%/20%
            X_train2, X_test2 = model_selection.train_test_split(cnv_sel_data, test_size=0.2, random_state=1)
            X_train3, X_test3 = model_selection.train_test_split(miRNA_sel_data, test_size=0.2, random_state=1)

            sae=   MutiViewAutoencoder2(X_train2,X_train3,X_test2,X_test3, activation = act  )
            trials = Trials()
            #define the space of hyperparameters
            space = {
                     'units1': hp.choice('units1', range(1,n_hidden2)),
                     'units2': hp.choice('units2', range(1,n_hidden3)),
                     'batch_size': hp.choice('batch_size',[16,8,4] ) ,
                     'alpha' :hp.choice('alpha',[0, hp.uniform('alpha2',0,1)]),
                     'learning_rate' : hp.loguniform('learning_rate',-5,-1),
                     'lamda':hp.choice('lamda',[0,hp.loguniform('lamda2',-8,-1)]),
                     'optimizer':hp.choice('optimizer',["adam","nadam","SGD","Momentum","RMSProp"]),
                     'initializer':hp.choice('initializer',["normal","uniform","xavier","He"]),
                     'keep_prob':hp.choice('keep_prob',[1, hp.uniform('kp',0.5,1)])

                    }        
        
        if miRNA_nbr <2 and cnv_nbr >1 and mrna_nbr>1:
            #Split dataset to training and test data 80%/20%
            X_train1, X_test1 = model_selection.train_test_split(mrna_sel_data, test_size=0.2, random_state=1)
            X_train2, X_test2 = model_selection.train_test_split(cnv_sel_data, test_size=0.2, random_state=1)

            sae=   MutiViewAutoencoder2(X_train1,X_train2,X_test1,X_test2, activation = act  )
            trials = Trials()
            #define the space of hyperparameters
            space = {
                     'units1': hp.choice('units1', range(1,n_hidden1)),
                     'units2': hp.choice('units2', range(1,n_hidden2)),
                     'batch_size': hp.choice('batch_size',[16,8,4] ) ,
                     'alpha' :hp.choice('alpha',[0, hp.uniform('alpha2',0,1)]),
                     'learning_rate' : hp.loguniform('learning_rate',-5,-1),
                     'lamda':hp.choice('lamda',[0,hp.loguniform('lamda2',-8,-1)]),
                     'optimizer':hp.choice('optimizer',["adam","nadam","SGD","Momentum","RMSProp"]),
                     'initializer':hp.choice('initializer',["normal","uniform","xavier","He"]),
                     'keep_prob':hp.choice('keep_prob',[1, hp.uniform('kp',0.5,1)])

                    }
        if mrna_nbr >1 and cnv_nbr >1 and miRNA_nbr>1:
            #Split dataset to training and test data 80%/20%
            X_train1, X_test1 = model_selection.train_test_split(mrna_sel_data, test_size=0.2, random_state=1)
            X_train2, X_test2 = model_selection.train_test_split(cnv_sel_data, test_size=0.2, random_state=1)
            X_train3, X_test3 = model_selection.train_test_split(miRNA_sel_data, test_size=0.2, random_state=1)

            sae=   MutiViewAutoencoder(X_train1,X_train2,X_train3,X_test1,X_test2,X_test3, activation = act  )
            trials = Trials()
            #define the space of hyperparameters
            space = {
                     'units1': hp.choice('units1', range(1,n_hidden1)),
                     'units2': hp.choice('units2', range(1,n_hidden2)),
                     'units3': hp.choice('units3', range(1,n_hidden3)),
                     'batch_size': hp.choice('batch_size',[16,8,4] ) ,
                     'alpha' :hp.choice('alpha',[0, hp.uniform('alpha2',0,1)]),
                     'learning_rate' : hp.loguniform('learning_rate',-5,-1),
                     'lamda':hp.choice('lamda',[0,hp.loguniform('lamda2',-8,-1)]),
                     'optimizer':hp.choice('optimizer',["adam","nadam","SGD","Momentum","RMSProp"]),
                     'initializer':hp.choice('initializer',["normal","uniform","xavier","He"]),
                     'keep_prob':hp.choice('keep_prob',[1, hp.uniform('kp',0.5,1)])
                     
                    }
        if mrna_nbr >1 or cnv_nbr >1 or miRNA_nbr>1:

#        train the HP optimization with 50 iterations
          best = fmin(sae.cross_validation, space, algo=tpe.suggest, max_evals=50,trials=trials)
          fname = './COAD/COAD_trials_HPMV_maxepochs1000_max_evals50_tanh2.pkl'
          pickle.dump(trials,open( fname, "ab" ))
            
#        get the loss of training the model on test data
          loss= sae.train_test(hyperopt.space_eval(space, best))
          print(best)
#        save the best HPs

          f= open('./COAD/COAD_MVGBMparameter_tanh2.txt', 'a+')
          for k, v in best.items():
            f.write(str(v) +',')
          f.write(str(loss))
          f.write('\n')
          f.close()
#

#Release memory
          del(mrna_sel_data)
          del(cnv_sel_data)
          del(mirna_sel_data)          
          del(trials)
          del(sae)


