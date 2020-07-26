import abc
import math
import numpy as np
#import tensorflow as tf


class MomentumSchedule(object):
    
    @abc.abstractmethod
    def __call__(self, step):
        
        raise NotImplementedError('Momentum schedule must override __call__')

    @abc.abstractmethod
    def get_config(self):
        
        raise NotImplementedError('Momentum schedule must override get_config')

    @classmethod
    def from_config(cls, config):
        
        """
        Instantiates a 'MomentumSchedule' from its config.
        
        Args:
            config: Output of 'get_config()'.
        
        Returns:
            A 'MomentumSchedule' instance.
        """

        return cls(**config)


class NesterovScheduler(MomentumSchedule):
    
    def __init__(self, momentum, name=None):
        
        super(NesterovScheduler, self).__init__()

        self.momentum = momentum
        self.name = name

    def __call__(self, step):

        m = 1 - 2**(-1-math.log2(np.floor(step/250)+1))
        m_t = min(m, self.momentum)

        return m_t

    def get_config(self,):

        return {
            'max_momentum': self.momentum,
            'name': self.name
        }
