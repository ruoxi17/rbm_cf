import numpy as np
import pandas as pd
import tensorflow as tf

class RBM:
  
    def __init__(self, 
                 n_visible, 
                 n_hidden, 
                 k=256, 
                 sample_fn=None,
                 err_fn=None, 
                 lr=0.1, 
                 num_epochs=32, 
                 batch_size=128, 
                 momentum=0.95,
                 normalized=False
              ):
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.sample_fn = sample_fn
        self.err_fn = err_fn
        self.k = k
        self.lr = lr
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.momentum = momentum
        self.normalized = normalized
        
        self.v = None
        self.W = None
        self.b_h = None
        self.b_v = None
        
        self.delta_w = None
        self.delta_bv = None
        self.delta_bh = None
        
        self.v_hat = None
        self.h_hat = None
        
        self.updates = None
        
        self.err = 0
        self.errs = 0
        
        assert self.sample_fn != None
        assert self.err_fn != None
        self._build_graph()
        
        init = tf.compat.v1.global_variables_initializer()
        self.sess = tf.compat.v1.Session()
        self.sess.run(init)
        
        writer = tf.compat.v1.summary.FileWriter('./graphs', self.sess.graph)
        
   
    def _build_graph(self):
        
        self.v = tf.compat.v1.placeholder(tf.float32, shape=[None, self.n_visible], name='v')
        
        xavier_init = tf.contrib.layers.xavier_initializer()
        with tf.compat.v1.variable_scope('variables'):
            self.W = tf.Variable(xavier_init([self.n_visible, self.n_hidden]), name='W')
            self.b_h = tf.Variable(tf.zeros([self.n_hidden], dtype=tf.float32), name='b_h')
            self.b_v = tf.Variable(tf.zeros([self.n_visible], dtype=tf.float32), name='b_v')
            with tf.compat.v1.variable_scope('deltas'):
                self.delta_w = tf.Variable(tf.zeros([self.n_visible, self.n_hidden]), dtype=tf.float32, name='delta_W')
                self.delta_bv = tf.Variable(tf.zeros([self.n_visible]), dtype=tf.float32, name='delta_bv')
                self.delta_bh = tf.Variable(tf.zeros([self.n_hidden]), dtype=tf.float32, name='delta_bh')
        
        self._params_updates_graph()
        self._predicts_graph()
        self._err_graph()
    
    
    def _params_updates_graph(self):
        
        with tf.compat.v1.variable_scope('gradients'):
            hidden_prob = tf.nn.sigmoid(tf.matmul(self.v, self.W) + self.b_h)
            visible_recon_prob = tf.nn.sigmoid(tf.matmul(self.sample_fn(hidden_prob), tf.transpose(self.W)) + self.b_v)
            hidden_recon_prob = tf.nn.sigmoid(tf.matmul(visible_recon_prob, self.W) + self.b_h)

            positive_grad = tf.matmul(tf.transpose(self.v), hidden_prob)
            negative_grad = tf.matmul(tf.transpose(visible_recon_prob), hidden_recon_prob)

            d_w = self.f(self.delta_w, positive_grad - negative_grad)
            d_bv = self.f(self.delta_bv, tf.reduce_mean(self.v - visible_recon_prob, 0))
            d_bh = self.f(self.delta_bh, tf.reduce_mean(hidden_prob - hidden_recon_prob, 0))

            self.updates = [self.delta_w.assign(d_w), self.delta_bv.assign(d_bv), self.delta_bh.assign(d_bh), self.W.assign_add(d_w), self.b_v.assign_add(d_bv), self.b_h.assign_add(d_bh)]
        
    
    def _predicts_graph(self):
        
        with tf.compat.v1.variable_scope('prediction'):
            self.h_hat = tf.nn.sigmoid(tf.matmul(self.v, self.W) + self.b_h)
            self.v_hat = tf.nn.sigmoid(tf.matmul(self.h_hat, tf.transpose(self.W)) + self.b_v)
        
        
    def _err_graph(self):
        
        with tf.compat.v1.variable_scope('errors'):
            h_hat = tf.nn.sigmoid(tf.matmul(self.v, self.W) + self.b_h)
            v_hat = tf.nn.sigmoid(tf.matmul(h_hat, tf.transpose(self.W)) + self.b_v)
            self.err = self.err_fn(self.v, v_hat)
    
    
    def get_errs(self):
        assert len(self.errs) > 0
        return self.errs[1:]
    
    
    def get_weights(self):
        return self.sess.run(self.W), self.sess.run(self.b_v), self.sess.run(self.b_h)
    
    
    def fit(self, data):
        
        if not self.normalized:
            data = (data - data.min()) / (data.max() - data.min())
        total_batch = int(len(data) / self.batch_size)

        for epoch in range(self.num_epochs):
            for i in range(total_batch):
                batch_v = data[i * self.batch_size : (i + 1) * self.batch_size]
                ret = self.sess.run([self.updates, self.err], feed_dict={self.v: batch_v.reshape((-1, self.n_visible))})
                self.errs = np.hstack((self.errs, ret[-1]))
            print('Epoch: %04d, err=%.4f' % (epoch + 1, ret[-1]))
        
        return self.get_errs()

    
    def reconstruct(self, batch_v):
        return self.sess.run(self.v_hat,feed_dict={self.v: batch_v.reshape((-1, self.n_visible))})

    
    def f(self, x_old, x_new):
        return self.momentum * x_old + self.lr * x_new * (1 - self.momentum) / tf.cast(tf.shape(x_new)[0], tf.float32)



    
