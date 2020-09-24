import abc
import tensorflow as tf


def log(x, base, dtype=tf.float32):
    """
    Logarithm with base.

    Args:
        x: A `Tensor`.
        base: A `Tensor`.

    Returns:
        A `Tensor` with same shape as x.
    """
    n = tf.math.log(tf.cast(x, dtype))
    d = tf.math.log(tf.cast(base, dtype))

    return tf.divide(n, d)


class MomentumSchedule:
    """
    A serializable momentum schedule.

    When training a model, a MomentumSchedule can be passed in as the momentum
    of the Momentum optimizer, see optimizers.py.
    """

    @abc.abstractmethod
    def __call__(self, step):
        raise NotImplementedError("Momentum schedule must override __call__")

    @abc.abstractmethod
    def get_config(self):
        raise NotImplementedError("Momentum schedule must override get_config")

    @classmethod
    def from_config(cls, config):
        """
        Instantiates a MomentumSchedule class from its config.

        Arguments:
            config: Output of `get_config()`.

        Returns:
            A MomentumSchedule instance.
        """
        return cls(**config)


class ConvexSchedule(MomentumSchedule):
    """
    A MomentumSchedule suitable for convex functions.
    See http://proceedings.mlr.press/v28/sutskever13.pdf.

    ```python
    schedule = schedules.ConvexSchedule()
    model.compile(
        optimizer=optimizers.Momentum(momentum=schedule),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    model.fit(data, labels, epochs=5)
    ```

    Args:
        const: A scalar `float32` or `float64` `Tensor` or a Python number.
            Defaults to 1.0. The multiplicative constant.
        name: String. Defaults to `ConvexSchedule`.
            Optional name of the operation.

    Returns:
        A 1-arg callable momentum schedule that takes the current optimizer step
        and outputs a scalar `Tensor` as momentum value.
    """

    def __init__(self, const: float = 1.0, name: str = None):
        super(ConvexSchedule, self).__init__()

        if const < 0 or const > 1:
            raise ValueError("`const` must be between [0, 1].")
        self.const = const

        self.name = name

    def __call__(self, step):
        with tf.name_scope(self.name or "ConvexSchedule") as name:
            const = tf.convert_to_tensor(
                self.const, dtype=tf.float32, name="multiplicative_constant"
            )
            step = tf.cast(step, tf.float32)
            step_value = const * 3 / (step + 5)

            return tf.subtract(1, step_value, name=name)

    def get_config(self):
        return {"const": self.const, "name": self.name}


class StronglyConvexSchedule(MomentumSchedule):
    """
    A MomentumSchedule achieving exponential convergence on
    strongly convex functions, see http://proceedings.mlr.press/v28/sutskever13.pdf.

    ```python
    schedule = schedules.StronglyConvexSchedule(upper_momentum=0.99,)
    model.compile(
        optimizer=optimizers.Momentum(momentum=schedule),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    model.fit(data, labels, epochs=5)
    ```

    Args:
        upper_momentum:  A scalar `float32` or `float64` `Tensor` or a
            Python number. Defaults to 0.99. The upper momentum.
        name: String. Defaults to `StronglyConvexSchedule`.
            Optional name of the operation.

    Returns:
        A 1-arg callable momentum schedule that takes the current
        optimizer step and outputs the momentum value, a scalar `Tensor` of
        the same type as `upper_momentum`.
    """

    def __init__(self, upper_momentum: float = 0.99, name: str = None):
        super(StronglyConvexSchedule, self).__init__()

        if upper_momentum < 0 or upper_momentum > 1:
            raise ValueError("`upper_momentum` must be between [0, 1].")
        self.upper_momentum = upper_momentum

        self.name = name

    def __call__(self, step):
        with tf.name_scope(self.name or "StronglyConvexSchedule") as name:
            upper_momentum = tf.convert_to_tensor(
                self.upper_momentum, dtype=tf.float32, name="upper_momentum"
            )
            step = tf.cast(step, tf.float32)
            step_value = 1 - tf.pow(2.0, -1 - log(tf.floor(step / 250) + 1, 2))

            return tf.minimum(step_value, upper_momentum, name=name)

    def get_config(self):
        return {"upper_momentum": self.upper_momentum, "name": self.name}
