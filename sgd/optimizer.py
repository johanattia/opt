import tensorflow as tf
from tensorflow.python.training.tracking import base as trackable

from schedule import MomentumSchedule


class Momentum(tf.keras.optimizers.Optimizer):
    """Stochastic gradient descent and momentum optimizer, including Nesterov acceleration.
    
    This implementation allows momentum schedule, see schedule module. Concerning formulation details, see :
    * http://www.cs.toronto.edu/~hinton/absps/momentum.pdf
    * https://arxiv.org/pdf/1212.0901.pdf
    """
    
    def __init__(self, 
                 learning_rate: float = 0.01, 
                 momentum: float = 0.0, 
                 nesterov: bool = False, 
                 name: str = 'Momentum',
                 **kwargs):
        """
        Construct a new Stochastic Gradient Descent or Momentum optimizer.

        Args:
            learning_rate: float hyperparameter >= 0. 
                Learning rate.
            momentum: float hyperparameter >= 0 that accelerates SGD in the relevant
                direction and dampens oscillations.
            nesterov:
            name:
        """
        super(Momentum, self).__init__(name, **kwargs)
        self._set_hyper('learning_rate', kwargs.get('lr', learning_rate))
        self._set_hyper('decay', self._initial_decay)
        self._set_hyper_momentum('momentum', momentum)
        self.nesterov = nesterov
        
    def _set_hyper(self, name, value):
        """set hyper `name` to value. value can be callable, tensor, numeric."""
        if isinstance(value, trackable.Trackable):
            self._track_trackable(value, name, overwrite=True)
        if name not in self._hyper:
            self._hyper[name] = value
        else:
            prev_value = self._hyper[name]
            if (callable(prev_value)
                or isinstance(prev_value, (tf.Tensor, int, float, tf.keras.optimizers.schedules.LearningRateSchedule))
                or isinstance(value, tf.keras.optimizers.schedules.LearningRateSchedule)):
                self._hyper[name] = value
            else:
                tf.keras.backend.set_value(self._hyper[name], value)
    
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
    
    def _create_slots(self, var_list):
        for var in var_list:
            self.add_slot(var, 'momentum', initializer='zeros')
            
    def _resource_apply_dense(self, grad, var, apply_state=None):
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
        
    def _resource_apply_sparse(self, grad, var, indices, **kwargs):
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
        config = super(Momentum, self).get_config()
        config.update({
            'learning_rate': self._serialize_hyperparameter('learning_rate'),
            'decay': self._serialize_hyperparameter('decay'),
            'momentum': self._serialize_hyperparameter('momentum'),
            'nesterov': self.nesterov,
        })
        return config