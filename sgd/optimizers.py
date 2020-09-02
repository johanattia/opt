from typing import Union

import tensorflow as tf
from tensorflow.python.training.tracking import base as trackable

from schedule import MomentumSchedule


# TODO : minimizer function for TensorFlow


class MomentumScheduledSGD(tf.keras.optimizers.Optimizer):  # TODO : _prepare_local
    """
    Stochastic gradient descent and momentum optimizer, including Nesterov acceleration.

    This implementation allows momentum schedule, see schedule.py. Concerning formulation details, see :
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

        super(MomentumScheduledSGD, self).__init__(name, **kwargs)
        self._set_hyper("learning_rate", kwargs.get("lr", learning_rate))
        self._set_hyper("decay", self._initial_decay)

        self._momentum = False
        if (
            isinstance(momentum, (tf.Tensor, MomentumSchedule))
            or callable(momentum)
            or momentum > 0
        ):
            self._momentum = True
        
        if isinstance(momentum, (int, float)) and (momentum < 0 or momentum > 1):
            raise ValueError("`momentum` must be between [0, 1].")

        self._set_hyper("momentum", momentum)
        self.nesterov = nesterov

    def _set_hyper(self, name, value):
        """Set hyper `name` to value. value can be callable, tensor, numeric.
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
        """Get value of hyper `name`. value can be callable, tensor, numeric.
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
        """Initialize new variables for momentum optimizer. Default initializer generates
        tensors initialized to 0.
        """
        if self._momentum:
            for var in var_list:
                self.add_slot(var, "momentum")

    def _prepare_local(self, var_device, var_dtype, apply_state):
        """docstring..."""
        # TODO : dynamic momentum
        super(MomentumScheduledSGD, self)._prepare_local(
            var_device, var_dtype, apply_state
        )
        apply_state[(var_device, var_dtype)]["momentum"] = tf.identity(
            self._get_hyper("momentum", var_dtype)
        )

    def _resource_apply_dense(self, grad, var, apply_state=None):
        # var_dtype = var.dtype.base_dtype
        #
        # lr_t = self._decayed_lr(var_dtype)
        # m_t = self._get_hyper('momentum', var_dtype)
        #
        # v = self.get_slot(var, 'momentum')
        # v_t = v.assign(m_t * v - lr_t * grad, use_locking=self._use_locking)
        #
        # if self.nesterov:
        #    var_update = var.assign_add(- m_t * v + (1 + m_t) * v_t, use_locking=self._use_locking)
        #
        # else:
        #    var_update = var.assign_add(v_t, use_locking=self._use_locking)
        #
        # return tf.group(*[var_update, v_t])
        raise NotImplementedError("Not yet implemented")

    def _resource_apply_sparse(self, grad, var, indices, **kwargs):
        # var_dtype = var.dtype.base_dtype
        #
        # lr_t = self._decayed_lr(var_dtype)
        # m_t = self._get_hyper('momentum', var_dtype)
        #
        # v = self.get_slot(var, 'momentum')
        # v_t = v.assign(m_t * v - lr_t * grad, use_locking=self._use_locking)
        #
        #
        #
        # return tf.group(*[var_update, v_t])
        raise NotImplementedError("Not yet implemented")

    def _serialize_hyperparameter(self, hyperparameter_name):
        """Serialize a hyperparameter that can be a float, callable, or Tensor.
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
        config = super(MomentumScheduledSGD, self).get_config()
        config.update(
            {
                "learning_rate": self._serialize_hyperparameter("learning_rate"),
                "decay": self._serialize_hyperparameter("decay"),
                "momentum": self._serialize_hyperparameter("momentum"),
                "nesterov": self.nesterov,
            }
        )
        return config