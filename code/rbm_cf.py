import numpy as np
import pandas as pd
import tensorflow as tf
import preprocessing as pre

class RBM:
  
    def __init__(self, 
                 n_visible, 
                 n_hidden, 
                 m, 
                 indices=None, 
                 sample_fn=None,
                 err_fn=None, 
                 lr=0.1, 
                 num_epochs=16, 
                 batch_size=64, 
                 momentum=0.95,
                 one_hot=False
              ):
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.m = m
        self.sample_fn = sample_fn
        self.err_fn = err_fn
        self.indices = indices
        self.lr = lr
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.momentum = momentum
        self.one_hot = one_hot
        
        self.v = []
        self.W = None # W_i,j,k
        self.b_h = None
        self.b_v = None # bv_i,k
        
        self.delta_w = None
        self.delta_bv = None
        self.delta_bh = None
        
        self.v_hat = None # v_i,k
        self.h_hat = None
        
        self.updates = []
        
        self.err = 0
        self.errs = 0
        
        assert self.indices != None
        assert self.sample_fn != None
        assert self.err_fn != None
        
        self._build_graph()
        
        init = tf.compat.v1.global_variables_initializer()
        self.sess = tf.compat.v1.Session()
        self.sess.run(init)
        
        writer = tf.compat.v1.summary.FileWriter('../graphs', self.sess.graph)
        
   
    def _build_graph(self):
        
        self._var_graph()
        self._params_updates_graph()
        #self._predicts_graph()
        #self._err_graph()
    
    def _var_graph(self):
        
        self.v = tf.compat.v1.placeholder(tf.float32, shape=[None, self.m, self.n_visible])
        
        xavier_init = tf.contrib.layers.xavier_initializer()

        with tf.compat.v1.variable_scope('variables'):
            self.W = tf.Variable(xavier_init([self.m, self.n_visible, self.n_hidden]))
            self.b_v = tf.Variable(tf.zeros([self.m, self.n_visible], dtype=tf.float32))
            self.delta_w = tf.Variable(tf.zeros([self.m, self.n_visible, self.n_hidden]), dtype=tf.float32)
            self.delta_bv = tf.Variable(tf.zeros([self.m, self.n_visible]), dtype=tf.float32)

            self.b_h = tf.Variable(tf.zeros([self.n_hidden], dtype=tf.float32), name='b_h')
            self.delta_bh = tf.Variable(tf.zeros([self.n_hidden]), dtype=tf.float32, name='delta_bh')
            
            
    def _params_updates_graph(self):
        
        with tf.compat.v1.variable_scope('gradients'):
            
            for i, index in enumerate(self.indices):
                
                print('user %d/%d' % (i+1, len(self.indices)))
                hidden_prob = []
                visible_recon_prob = []
                hidden_recon_prob = []

                # P(h^t|v), hidden layer is shared
                v = tf.gather(self.v, index, axis=1)
                for i_, idx in enumerate(index):
                    hidden_prob.append(tf.matmul(v[:, i_, :], self.W[idx, :, :]))
                hidden_prob = tf.sigmoid(tf.reduce_sum(hidden_prob, axis=0) + self.b_h) #[batch_size, n_hidden]
                
                with tf.compat.v1.variable_scope('prediction'):
                    v_hat = []
                    for i_ in range(self.n_visible):
                        temp = tf.exp(tf.matmul(self.sample_fn(hidden_prob), tf.transpose(self.W[:, i_, :])) + self.b_v[:, i_]) #[batch_size, M]
                        v_hat.append(tf.transpose(temp))
                    v_hat = tf.transpose(tf.stack(v_hat, axis=0))
                    self.v_hat = tf.exp(v_hat) / tf.reduce_sum(tf.exp(v_hat), axis=2, keepdims=True)
                        
                # P(v^(t+1) | h^t), separate v for single user, softmax over n_visible
                visible_recon_prob =  []
                W = tf.gather(self.W, index)
                b_v = tf.gather(self.b_v, index)
                for i_ in range(self.n_visible):
                    temp = tf.exp(tf.matmul(self.sample_fn(hidden_prob), tf.transpose(W[:, i_, :])) + b_v[:, i_]) #[batch_size, m]
                    visible_recon_prob.append(tf.transpose(temp))
                visible_recon_prob = tf.transpose(tf.stack(visible_recon_prob, axis=0))
                visible_recon_prob = tf.exp(visible_recon_prob) / tf.reduce_sum(tf.exp(visible_recon_prob), axis=2, keepdims=True)

                # P(h^(t+1) | v^(t+1))
                visible_recon_samp = self.sample_fn(visible_recon_prob)
                
                with tf.compat.v1.variable_scope('errors'):
                    self.err = self.err_fn(v, visible_recon_prob) #prob errors
                    # self.err = self.err_fn(self.v[i], visible_recon_samp) # samp errors
                    
                for i_, idx in enumerate(index):
                    hidden_recon_prob.append(tf.matmul(visible_recon_samp[:, i_, :], self.W[idx, :, :]))
                hidden_recon_prob = tf.sigmoid(tf.reduce_sum(hidden_recon_prob, axis=0) + self.b_h)

                # update gradients and variables
                d_W = []
                d_bv = []
                curr = 0
                
                for i_ in range(self.m):
                    # if i in indices, matmul; else append zeros
                    if i_ in index:
                        positive_grad = tf.matmul(tf.transpose(v[:, curr, :]), hidden_prob)
                        negative_grad = tf.matmul(tf.transpose(visible_recon_prob[:, curr, :]), hidden_recon_prob)
                
                        d_W.append(self.f(self.delta_w[i_, :, :], positive_grad - negative_grad))
                        d_bv.append(self.f(self.delta_bv[i_, :], tf.reduce_mean(v[:, curr, :] - visible_recon_prob[:, curr, :], 0)))
                        curr += 1
                    else:
                        d_W.append(tf.zeros([self.n_visible, self.n_hidden]))
                        d_bv.append(tf.zeros([self.n_visible]))
                
                d_bh = self.f(self.delta_bh, tf.reduce_mean(hidden_prob - hidden_recon_prob, 0))
                d_W = tf.stack(d_W, axis=0)
                d_bv = tf.stack(d_bv, axis=0)
            
                self.updates.append([self.delta_w.assign(d_W), self.delta_bv.assign(d_bv), self.delta_bh.assign(d_bh), self.W.assign_add(d_W), self.b_v.assign_add(d_bv), self.b_h.assign_add(d_bh)])
            
    
    def get_errs(self):
        assert len(self.errs) > 0
        return self.errs[1:]
    
    
    def get_weights(self):
        return self.sess.run(self.W), self.sess.run(self.b_v), self.sess.run(self.b_h)
    
    
    def fit(self, data):

        for epoch in range(self.num_epochs):
            
            for i, batch_v in enumerate(data): # one user 
                
                err_sum = 0
                if not self.one_hot:
                    batch_v = pre.to_knary_one_hot(batch_v, self.n_visible)
                
                ret = self.sess.run([self.updates[i], self.err], feed_dict={self.v: batch_v})
                err_sum += ret[-1]
                
            self.errs = np.hstack((self.errs, err_sum / len(data)))    
            print('Epoch: %04d, err=%.8f' % (epoch + 1, self.errs[-1]))
        
        return self.get_errs()

    
    def reconstruct(self, indices, data):
        rets = []
        for i, idx in enumerate(indices):
            if not self.one_hot:
                batch_v = pre.to_knary_one_hot(data[i], self.n_visible)
            ret = self.sess.run(self.v_hat,feed_dict={self.v: batch_v})
            rets.append(ret.reshape(self.m, self.n_visible))
        return np.array(rets)
                
    
    def f(self, x_old, x_new):
        return self.momentum * x_old + self.lr * x_new * (1 - self.momentum) / tf.cast(tf.shape(x_new)[0], tf.float32)



    
