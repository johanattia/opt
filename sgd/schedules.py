import abc
import tensorflow as tf


def log(x, base):
    """Logarithm with base.
    
    Args:
        x: A `Tensor`.
        base: A `Tensor`.
    
    Returns:
        A `Tensor` with same shape as x.
    """

    n = tf.math.log(x)
    d = tf.math.log(tf.cast(base, dtype=n.dtype))

    return tf.divide(n, d)


class MomentumSchedule(object):
    """Abstract class for momentum schedule.
    
    An instance subclass of `MomentumSchedule`can be passed in as the momentum of `MomentumScheduledSGD`, see optimizers.py.
    """

    @abc.abstractmethod
    def __call__(self, step):
        raise NotImplementedError('Momentum schedule must override __call__')

    @abc.abstractmethod
    def get_config(self):
        raise NotImplementedError('Momentum schedule must override get_config')

    @classmethod
    def from_config(cls, config):
        """Instantiates a `MomentumSchedule` from its config.
        
        Args:
            config: Output of `get_config()`.
        
        Returns:
            A `MomentumSchedule` instance.
        """

        return cls(**config)


class ConvexSchedule(MomentumSchedule):
    """Momentum schedule suitable for convex functions, see http://www.cs.toronto.edu/~hinton/absps/momentum.pdf.

    ```python
    schedule = schedules.ConvexSchedule()
    model.compile(
        optimizer=optimizers.MomentumScheduledSGD(momentum=schedule),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    model.fit(data, labels, epochs=5)
    ```

    Returns:
        A 1-arg callable momentum schedule that takes the current optimizer step and outputs a scalar `Tensor` as momentum value.
    """
    
    def __init__(self, const=1.0, name=None):
        """Applies appropriate schedule on convex functions to the momentum.

        Args:
            const: A scalar `float32` or `float64` `Tensor` or a Python number. Defaults to 1.0.
                The multiplicative constant.
            
            name: String. Defaults to `ConvexSchedule`.
                Optional name of the operation.
        """

        super(ConvexSchedule, self).__init__()
        self.const = const
        self.name = name

    def __call__(self, step):

        with tf.name_scope(self.name or 'ConvexSchedule') as name:
            const = tf.convert_to_tensor(
                self.const,
                name='multiplicative_constant'
            )
            dtype = const.dtype
            step = tf.convert_to_tensor(step, dtype=dtype)
            step_value = tf.multiply(const, tf.divide(3, tf.add(step, 5)))

            return tf.subtract(1, step_value, name=name)

    def get_config(self):
        return {
            'const': self.const,
            'name': self.name
        }


class StronglyConvexSchedule(MomentumSchedule):
    """Momentum schedule achieving exponential convergence on strongly convex functions, see http://www.cs.toronto.edu/~hinton/absps/momentum.pdf.

    ```python
    schedule = schedules.StronglyConvexSchedule(upper_momentum=0.99,)
    model.compile(
        optimizer=optimizers.MomentumScheduledSGD(momentum=schedule),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    model.fit(data, labels, epochs=5)
    ```

    Returns:
        A 1-arg callable momentum schedule that takes the current optimizer step and outputs the momentum value, 
        a scalar `Tensor` of the same type as `upper_momentum`.
    """
    
    def __init__(self, upper_momentum=0.99, name=None):
        """Applies schedule achieving exponential convergence on strongly convex functions to the momentum.

        Args:
            upper_momentum:  A scalar `float32` or `float64` `Tensor` or a Python number. Defaults to 0.99.
                The upper momentum.
                
            name: String. Defaults to `StronglyConvexSchedule`.
                Optional name of the operation.
        """

        super(StronglyConvexSchedule, self).__init__()
        self.upper_momentum = upper_momentum
        self.name = name

    def __call__(self, step):

        with tf.name_scope(self.name or 'StronglyConvexSchedule') as name:
            upper_momentum = tf.convert_to_tensor(
                self.upper_momentum, 
                name='upper_momentum'
            )
            dtype = upper_momentum.dtype
            step = tf.cast(step, dtype=dtype)
            step_value = 1 - tf.pow(2., -1 - log(tf.floor(step/250) + 1, 2))

            return tf.minimum(step_value, upper_momentum, name=name)

    def get_config(self):
        return {
            'upper_momentum': self.upper_momentum,
            'name': self.name
        }
