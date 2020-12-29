from typing import Callable, Optional
import tensorflow as tf


def minimizer(
    func: Callable,
    optimizer: tf.keras.optimizers.Optimizer,
    initial_position: tf.Variable,
    verbose: bool = False,
    name: Optional[str] = None,
):
    if not callable(func):
        raise TypeError("func must be a valid callable Python object.")
    if not isinstance(optimizer, tf.keras.optimizers.Optimizer):
        raise TypeError("optimizer must be a valid tf.keras.optimizers.Optimizer")

    if isinstance(initial_position, tf.Variable):
        comparing_position = tf.Variable(initial_position.value())
    else:
        initial_position = tf.Variable(initial_position)
        comparing_position = tf.Variable(initial_position)

    history = {"function_value": [], "position_value": []}
    iterations = tf.Variable(0)

    with tf.name_scope(name or "minimizer"):

        def _cond(initial_position, comparing_position):
            return tf.reduce_all(
                tf.less_equal(func(comparing_position), func(initial_position))
            )

        def _body(initial_position, comparing_position):
            initial_position.assign(comparing_position.value())
            cost_function = lambda: func(comparing_position)
            optimizer.minimize(cost_function, [comparing_position])

            history["function_value"].append(func(initial_position).numpy())
            history["position_value"].append(initial_position.numpy())
            iterations.assign_add(1)

            if verbose:
                tf.print(
                    "Iteration",
                    iterations,
                    "- Function value :",
                    history["function_value"][-1],
                    "- Position value :",
                    history["position_value"][-1],
                )
            return initial_position, comparing_position

        tf.while_loop(
            cond=_cond,
            body=_body,
            loop_vars=[initial_position, comparing_position],
            shape_invariants=[initial_position.shape, comparing_position.shape],
        )

        history["iterations"] = iterations.numpy()
        return history
