import tensorflow as tf

import numpy as np
import os
import time

class Pretrainer(object):

    def __init__(self, training_folder, checkpoint_dir, seq_length=100, files_extension='.py', embedding_dim=256,
                rnn_units=1024, BATCH_SIZE=64, BUFFER_SIZE=10000, EPOCHS=1000):
        self.training_folder = training_folder
        self.checkpoint_dir = checkpoint_dir
        self.files_extension = files_extension
        self.seq_length = seq_length
        self.embedding_dim = embedding_dim
        self.rnn_units = rnn_units
        self.BATCH_SIZE = BATCH_SIZE
        self.BUFFER_SIZE = BUFFER_SIZE
        self.EPOCHS = EPOCHS


    def prepare_data(self):
        # Read the training files
        self.text = ''
        for fname in os.listdir(self.training_folder):
            if fname.endswith(self.files_extension):
                with open(os.path.join(self.training_folder, fname), 'rb') as f:
                    self.text += f.read().decode(encoding='utf-8') + '\n'
        # Generate vocab char by char
        self.vocab = sorted(set(self.text))
        # Assign an index for each different char
        self.char2idx = {u:i+1 for i, u in enumerate(self.vocab)}
        self.char2idx[''] = 0
        # Map between indeces to char
        self.idx2char = {i+1:u for i, u in enumerate(self.vocab)}
        self.idx2char[0] = ''

        # Prepare the sequences
        text_as_int = np.array([self.char2idx[c] for c in self.text]) # Convert the text into indeces representation
        char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
        sequences = char_dataset.batch(self.seq_length+1, drop_remainder=True)

        def split_input_target(chunk):
            input_text = chunk[:-1]
            target_text = chunk[1:]
            return input_text, target_text

        # Split input and targets values
        self.dataset = sequences.map(split_input_target)
        self.dataset = self.dataset.shuffle(self.BUFFER_SIZE).batch(self.BATCH_SIZE, drop_remainder=True)
    
    
    def build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(len(self.vocab) + 1, self.embedding_dim,
                                    mask_zero=True, batch_input_shape=[self.BATCH_SIZE, None]),
            tf.keras.layers.GRU(self.rnn_units,
                                return_sequences=True,
                                stateful=True,
                                recurrent_initializer='glorot_uniform'),
            tf.keras.layers.Dense(len(self.vocab) + 1)
        ])
        return model

    
    def train(self):
        # Load dataset
        self.prepare_data()

        # Build model
        model = self.build_model()

        def loss(labels, logits):
            return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)


        model.compile(optimizer='adam', loss=loss)


        # Checkpoints callback

        # Name of the checkpoint files
        checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt_{epoch}")

        checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_prefix,
            save_weights_only=True,
            period=self.EPOCHS/10
        )

        model.fit(self.dataset, epochs=self.EPOCHS, callbacks=[checkpoint_callback])