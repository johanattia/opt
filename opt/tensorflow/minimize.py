from typing import Callable, Optional
import tensorflow as tf


# TODO : tf.while_loop


def minimizer(
    func: Callable,
    optimizer: tf.keras.optimizers.Optimizer,
    initial_position: tf.Variable,
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

    history = dict(value_function=[], antecedent=[])
    iterations = tf.Variable(0)

    with tf.name_scope(name or "minimize"):
        while tf.reduce_all(
            tf.less_equal(func(comparing_position), func(initial_position))
        ):
            initial_position.assign(comparing_position.value())

            with tf.GradientTape() as tape:
                value_function = func(comparing_position)

            grad = tape.gradient(value_function, [comparing_position])
            optimizer.apply_gradients(zip(grad, [comparing_position]))

            history["value_function"].append(func(initial_position).numpy())
            history["antecedent"].append(initial_position.numpy())
            iterations.assign_add(1)

            tf.print(
                "Iterations :",
                iterations,
                "- Value function :",
                history["value_function"][-1],
                "- Antecedent :",
                history["antecedent"][-1],
            )
        return history
