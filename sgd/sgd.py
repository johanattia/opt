import numpy as np
#import pandas as pd
import tensorflow as tf


class GradientDescentOptimizer(object):
    
    def __init__(self, func, initial_x, learning_rate=0.1, momentum=0, nesterov=False): # tolerance & patience parameters ?
        
        if callable(func):
            self.func = func
        
        else:
            raise TypeError("func object is not callable")
        
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.nesterov = nesterov
        
        self.iteration = 0

        # Initialization
        self.x = tf.Variable(initial_x)
        self.argmin = tf.Variable(initial_x)
        self.velocity = tf.zeros(self.x.shape)
        self.history = {'min': [], 'argmin': []}
     
    def optimize(self): # initial_x = tf.random.normal(shape=[2, 1], mean=0., stddev=1000.)
            
        # Iteration
        while self.func(self.x) <= self.func(self.argmin):
            
            self.argmin.assign(self.x)

            with tf.GradientTape() as tape: 
                value_func = self.func(self.x)

            gradient = tape.gradient(value_func, self.x)
            learning_rate = self.get_lr()
            momentum = self.get_momentum()
            v = momentum * self.velocity - learning_rate * gradient
            
            if self.nesterov:
                self.x.assign_add(- momentum * self.velocity + (1 + momentum) * v)
                
            else:
                self.x.assign_add(v)

            self.velocity = v
            self.history['min'].append(self.func(self.argmin).numpy())
            self.history['argmin'].append(self.argmin.numpy())
            
            print(f"Iteration : {self.iteration}. Value Function : {self.history['min'][-1]}.")
            print(f"argmin : {np.reshape(self.history['argmin'][-1], (2,))}\n")
            
            self.iteration += 1
            
        #self.history_df = pd.DataFrame(
        #    np.hstack([
        #        np.concatenate(self.history['argmin'], axis=1).T, 
        #        np.expand_dims(np.array(self.history['min']), axis=1)
        #    ]), 
        #    columns=['x', 'y', 'f(x,y)'],
        #)
        #self.history_df.index.name = 'Iteration'
        
        return
    
    def get_lr(self):
        
        if callable(self.learning_rate):
            return self.learning_rate(self.iteration)
        
        return self.learning_rate
                  
    def get_momentum(self):
                  
        if callable(self.momentum):
            return self.momentum(self.iteration)
                  
        return self.momentum



class StochasticGradientDescent(tf.keras.optimizers.Optimizer): # Add a method ._scheduled_momentum_
    
    """ Stochastic Gradient Descent implementation, including Momentum and Nesterov acceleration.
    
    References : 
    - https://arxiv.org/pdf/1212.0901.pdf
    - http://www.cs.toronto.edu/~hinton/absps/momentum.pdf
    - https://github.com/tensorflow/tensorflow/blob/v2.2.0/tensorflow/python/keras/optimizer_v2/optimizer_v2.py
    - https://github.com/tensorflow/tensorflow/blob/v2.1.0/tensorflow/python/keras/optimizer_v2/gradient_descent.py
    - https://github.com/tensorflow/addons/blob/master/tensorflow_addons/optimizers/rectified_adam.py
    - https://github.com/tensorflow/addons/blob/master/tensorflow_addons/optimizers/novograd.py
    
    """
    
    def __init__(self, learning_rate=0.01, momentum=0.0, nesterov=False, name='SGD', **kwargs):
        
        # Inheritance
        super(StochasticGradientDescent, self).__init__(name, **kwargs)
        
        # Hyperparameters
        self._set_hyper('learning_rate', kwargs.get('lr', learning_rate))
        self._set_hyper('decay', self._initial_decay)
        self._set_hyper('momentum', momentum)
        self.nesterov = nesterov
        
    def _create_slots(self, var_list):
        
        for var in var_list:
            self.add_slot(var, 'momentum', initializer='zeros')
            
        return
            
    def _resource_apply_dense(self, grad, var):
        
        var_dtype = var.dtype.base_dtype
        
        lr_t = self._decayed_lr(var_dtype)
        m_t = self._get_hyper('momentum', var_dtype)
        
        v = self.get_slot(var, 'momentum')
        v_t = v.assign(m_t * v - lr_t * grad, use_locking=self._use_locking)
        
        if self.nesterov:
            var_update = var.assign_add(- m_t * v + (1 + m_t) * v_t, use_locking=self._use_locking)
            
        else:
            var_update = var.assign_add(v_t, use_locking=self._use_locking)
            
        return tf.group(*[var_update, v_t])
        
    def _resource_apply_sparse(self, grad, var, indices):
        
    #    var_dtype = var.dtype.base_dtype
    #    
    #    lr_t = self._decayed_lr(var_dtype)
    #    m_t = self._get_hyper('momentum', var_dtype)
    #    
    #    v = self.get_slot(var, 'momentum')
    #    v_t = v.assign(m_t * v - lr_t * grad, use_locking=self._use_locking)
    #    
    #    
    #    
    #    return tf.group(*[var_update, v_t])

        raise NotImplementedError('Not yet implemented')
    
    def get_config(self):
        
        config = super(StochasticGradientDescent, self).get_config()
        config.update(
            {
                'learning_rate': self._serialize_hyperparameter('learning_rate'),
                'decay': self._serialize_hyperparameter('decay'),
                'momentum': self._serialize_hyperparameter('momentum'),
                'nesterov': self.nesterov,
            }
        )
        
        return config