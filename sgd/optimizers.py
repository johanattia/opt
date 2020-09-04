from typing import Union

import tensorflow as tf
from tensorflow.python.training.tracking import base as trackable

from schedule import MomentumSchedule


# TODO 1 : minimizer function for TensorFlow
# TODO 2 : minimizer function for TensorFlow
# TODO 3 : Momentum optimizer method : from_config


def resource_apply_scheduled_momentum(
    var: tf.Tensor,
    accum: tf.Tensor,
    lr: float,
    grad: tf.Tensor,
    current_momentum: float,
    next_momentum: float,
    use_locking: bool,
    use_nesterov: bool,
):

    return NotImplementedError  # tf.group(*updates)


def resource_sparse_apply_scheduled_momentum(
    var: tf.Tensor,
    accum: tf.Tensor,
    lr: float,
    grad: tf.Tensor,
    indices: tf.Tensor,
    current_momentum: float,
    next_momentum: float,
    use_locking: bool,
    use_nesterov: bool,
):

    return NotImplementedError  # tf.group(*updates)


class Momentum(tf.keras.optimizers.Optimizer):
    """
    Stochastic gradient descent and momentum optimizer, including Nesterov acceleration.

    This implementation allows momentum schedule, see schedule.py. Concerning formulation
    details, see :
    - http://proceedings.mlr.press/v28/sutskever13.pdf
    - https://arxiv.org/pdf/1212.0901.pdf

    Arguments:
        learning_rate: A `Tensor`, floating point value, or a schedule that is a
            `tf.keras.optimizers.schedules.LearningRateSchedule`, or a callable
            that takes no arguments and returns the actual value to use. The
            learning rate. Defaults to 0.01.
        momentum:  A `Tensor`, floating point value, or a schedule that is a
            `MomentumSchedule`, or a callable that takes no arguments and returns the
            actual value to use >= 0. Accelerates gradient descent in the relevant direction
            and dampens oscillations. Defaults to 0, i.e., vanilla gradient descent.
        nesterov: boolean. Whether to apply Nesterov momentum.
            Defaults to `False`.
        name: Optional name prefix for the operations created when applying
            gradients.  Defaults to `'SGD'`.
        **kwargs: Keyword arguments. Allowed to be one of
            `"clipnorm"` or `"clipvalue"`.
            `"clipnorm"` (float) clips gradients by norm; `"clipvalue"` (float) clips
            gradients by value.
    """

    def __init__(
        self,
        learning_rate: Union[
            float, tf.keras.optimizers.schedules.LearningRateSchedule
        ] = 0.01,
        momentum: Union[float, MomentumSchedule] = 0.0,
        nesterov: bool = False,
        name: str = "SGD",
        **kwargs
    ):
        super(Momentum, self).__init__(name, **kwargs)
        self._set_hyper("learning_rate", kwargs.get("lr", learning_rate))
        self._set_hyper("decay", self._initial_decay)

        self._momentum = False
        self._momentum_schedule = False
        if (
            isinstance(momentum, (tf.Tensor, MomentumSchedule))
            or callable(momentum)
            or momentum > 0
        ):
            self._momentum = True
        if isinstance(momentum, MomentumSchedule):
            self._momentum_schedule = True
        if isinstance(momentum, (int, float)) and (momentum < 0 or momentum > 1):
            raise ValueError("`momentum` must be between [0, 1].")
        self._set_hyper("momentum", momentum)

        self.nesterov = nesterov

    def _set_hyper(self, name, value):
        """
        Set hyper `name` to value. value can be callable, tensor, numeric.
        This method is overriden to allow MomentumSchedule use for momentum hyperparameter.
        """
        if isinstance(value, trackable.Trackable):
            self._track_trackable(value, name, overwrite=True)
        if name not in self._hyper:
            self._hyper[name] = value
        else:
            prev_value = self._hyper[name]
            if (
                callable(prev_value)
                or isinstance(
                    prev_value,
                    (
                        tf.Tensor,
                        int,
                        float,
                        tf.keras.optimizers.schedules.LearningRateSchedule,
                        MomentumSchedule,
                    ),
                )
                or isinstance(
                    value,
                    (
                        tf.keras.optimizers.schedules.LearningRateSchedule,
                        MomentumSchedule,
                    ),
                )
            ):
                self._hyper[name] = value
            else:
                tf.keras.backend.set_value(self._hyper[name], value)

    def _get_hyper(self, name, dtype=None):
        """
        Get value of hyper `name`. value can be callable, tensor, numeric.
        This method is overriden to allow MomentumSchedule use for momentum hyperparameter.
        """
        if not self._hypers_created:
            self._create_hypers()
        value = self._hyper[name]
        if isinstance(
            value,
            (tf.keras.optimizers.schedules.LearningRateSchedule, MomentumSchedule),
        ):
            return value
        if callable(value):
            value = value()
        if dtype:
            return tf.cast(value, dtype)
        else:
            return value

    def _create_slots(self, var_list):
        """
        Initialize new variables for momentum optimizer. Default initializer generates
        tensors initialized to 0.
        """
        if self._momentum:
            for var in var_list:
                self.add_slot(var, "momentum")

    def _prepare_local(self, var_device, var_dtype, apply_state):
        """
        Prepare learning rate and momentum as Tensors with dtype=var_dtype
        for var_device.
        """
        super(Momentum, self)._prepare_local(var_device, var_dtype, apply_state)

        momentum = self._get_hyper("momentum", var_dtype)
        if self._momentum_schedule:
            current_mt, next_mt = self._scheduled_momentum(momentum, var_dtype)
            apply_state[(var_device, var_dtype)].update(
                {"mt_t": tf.identity(current_mt), "mt_t+1": tf.identity(next_mt)}
            )
        else:
            apply_state[(var_device, var_dtype)]["momentum"] = tf.identity(momentum)

    def _scheduled_momentum(self, scheduler, var_dtype):
        """
        Get scheduled momentum states as Tensors with dtype=var_dtype.
        """
        local_step = tf.cast(self.iterations, var_dtype)
        current_mt = tf.cast(scheduler(local_step), var_dtype)
        next_mt = tf.cast(scheduler(local_step + 1), var_dtype)

        return current_mt, next_mt

    def _resource_apply_dense(self, grad, var, apply_state=None):
        """
        Update variable given gradient tensor is dense.
        """
        var_device, var_dtype = var.device, var.dtype.base_dtype
        coefficients = (apply_state or {}).get(
            (var_device, var_dtype)
        ) or self._fallback_apply_state(var_device, var_dtype)

        if self._momentum:
            momentum_var = self.get_slot(var, "momentum")
            if self._momentum_schedule:
                pass
            else:
                return tf.raw_ops.ResourceApplyKerasMomentum(
                    var=var.handle,
                    accum=momentum_var.handle,
                    lr=coefficients["lr_t"],
                    grad=grad,
                    momentum=coefficients["momentum"],
                    use_locking=self._use_locking,
                    use_nesterov=self.nesterov,
                )
        else:
            return tf.raw_ops.ResourceApplyGradientDescent(
                var.handle, coefficients["lr_t"], grad, use_locking=self._use_locking
            )

    def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
        """
        Update variable given gradient tensor is sparse.
        """
        # var_device, var_dtype = var.device, var.dtype.base_dtype
        # coefficients = (apply_state or {}).get(
        #    (var_device, var_dtype)
        # ) or self._fallback_apply_state(var_device, var_dtype)

        raise NotImplementedError("Not yet implemented")

    def _serialize_hyperparameter(self, hyperparameter_name):
        """
        Serialize a hyperparameter that can be a float, callable, or Tensor.
        This method is overriden to allow MomentumSchedule serialization.
        """
        value = self._hyper[hyperparameter_name]
        if isinstance(
            value,
            (tf.keras.optimizers.schedules.LearningRateSchedule, MomentumSchedule),
        ):
            return tf.keras.optimizers.schedules.serialize(value)
        if callable(value):
            return value()
        if tf.is_tensor(value):
            return tf.keras.backend.get_value(value)

        return value

    def get_config(self):
        config = super(Momentum, self).get_config()
        config.update(
            {
                "learning_rate": self._serialize_hyperparameter("learning_rate"),
                "decay": self._serialize_hyperparameter("decay"),
                "momentum": self._serialize_hyperparameter("momentum"),
                "nesterov": self.nesterov,
            }
        )
        return config
