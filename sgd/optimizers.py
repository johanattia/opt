import numpy as np
#import pandas as pd
import tensorflow as tf

from schedules import MomentumSchedule


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



class MomentumScheduledSGD(tf.keras.optimizers.Optimizer):
    """Stochastic gradient descent and momentum optimizer, including Nesterov acceleration.
    
    This implementation allows momentum scheduling. Concerning formulation details, see :
    - http://www.cs.toronto.edu/~hinton/absps/momentum.pdf
    - https://arxiv.org/pdf/1212.0901.pdf
    """
    
    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.0, nesterov: bool = False, name: str = 'SGD', **kwargs):

        """
        Construct a new stochastic gradient descent or momentum optimizer.

        Args:
            learning_rate:

            momentum:

            nesterov:

            name:
        """
        
        super(MomentumScheduledSGD, self).__init__(name, **kwargs)
        self._set_hyper('learning_rate', kwargs.get('lr', learning_rate))
        self._set_hyper('decay', self._initial_decay)
        self._set_hyper_momentum('momentum', momentum)
        self.nesterov = nesterov
        
    def _set_hyper_momentum(self, name, value):
        """Set hyper `name` to value. value can be callable, tensor, numeric or MomentumSchedule.
        
        This method is added for readability in MomentumSchedule use case.
        """

        if name not in self._hyper:
            self._hyper[name] = value

        else:
            prev_value = self._hyper[name]
            if (callable(prev_value) or isinstance(prev_value, (tf.Tensor, int, float, MomentumSchedule)) or isinstance(value, MomentumSchedule)):
                self._hyper[name] = value
            else:
                tf.keras.backend.set_value(self._hyper[name], value)

        return
    
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
    
    def _serialize_hyperparameter(self, hyperparameter_name):
        """Serialize a hyperparameter that can be a float, callable, or Tensor.
        
        This method is overriden to allow MomentumSchedule serialization, without loss of readability.
        """

        value = self._hyper[hyperparameter_name]
        if isinstance(value, (tf.keras.optimizers.schedules.LearningRateSchedule, MomentumSchedule)):
            return tf.keras.optimizers.schedules.serialize(value)
        
        if callable(value):
            return value()
        
        if tf.is_tensor(value):
            return tf.keras.backend.get_value(value)
        
        return value
    
    def get_config(self):

        config = super(MomentumScheduledSGD, self).get_config()
        config.update({
            'learning_rate': self._serialize_hyperparameter('learning_rate'),
            'decay': self._serialize_hyperparameter('decay'),
            'momentum': self._serialize_hyperparameter('momentum'),
            'nesterov': self.nesterov,
        })
        
        return config