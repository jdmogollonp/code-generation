from pretrainer.pretrainer import Pretrainer
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Pre-train the model
pretrainer = Pretrainer('./data/training/', '/Users/miller/Desktop/training_checkpoints')
pretrainer.prepare_data()
print(pretrainer.eval(10, 'def f(', 1.0, debug=True))