from collections import deque
import random
import numpy as np
import tensorflow as tf

class DDRQNAgent(object):
    def __init__(self, vocab_size, environment, checkpoint_dir, embedding_dim, rnn_units, output_file, batch_size=32, 
                temperature=1.0, render=False, memory_limit=20000, gamma=0.99, learning_rate=0.001, epsilon=0.2, 
                epsilon_min=0.01, epsilon_decay=0.95, episodes=10000):
        self.env = environment
        self.episodes = episodes
        self.temperature = temperature

        self.memory = deque(maxlen=memory_limit)

        self.gamma = gamma
        self.learning_rate = learning_rate

        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.train_network = self.create_network()

        self.target_network = self.create_network()
        self.target_network.set_weights(self.train_network.get_weights())
        
        self.vocab_size = vocab_size
        self.output_file = output_file
        self.batch_size = batch_size
        self.checkpoint_dir = checkpoint_dir
        self.embedding_dim = embedding_dim
        self.rnn_units = rnn_units
        
        # Render enviroment
        self.render = render

    
    def build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(len(self.vocab_size), self.embedding_dim,
                                    batch_input_shape=[self.batch_size, None]),
            tf.keras.layers.GRU(self.rnn_units,
                                return_sequences=False,
                                stateful=False,
                                recurrent_initializer='glorot_uniform'),
            tf.keras.layers.Dense(len(self.vocab_size))
        ])
        return model


    def create_network(self):
        tf.train.latest_checkpoint(self.checkpoint_dir)
        model = self.build_model()
        model.load_weights(tf.train.latest_checkpoint(self.checkpoint_dir))
        model.build(tf.TensorShape([self.batch_size, None]))
        model.summary()
        return model
    

    def choose_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randint(0, len(self.vocab_size))
        else:
            predictions = self.train_network(state)
            # remove the batch dimension
            predictions = tf.squeeze(predictions, 0)

            # using a categorical distribution to predict the character returned by the model
            predictions = predictions / self.temperature
            predicted_id = tf.random.categorical(predictions, num_samples=1).numpy()
            return predicted_id
    

    def remember(self, state, action, next_state, reward, done):
        self.memory.append([state, action, next_state, reward, done])


    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        # Get batch from memory
        mini_batch = random.sample(self.memory, self.batch_size)

        # Get states tables
        states = []
        next_states = []

        for state, action, new_state, reward, done in mini_batch:
            states.append(state)
            next_states.append(new_state)

        # Predict target values
        state_targets = self.train_network.predict(states)
        next_state_targets = self.target_network.predict(next_states)

        # Train neural network
        i = 0
        for state, action, new_state, reward, done in mini_batch:
            state_targets[i][action] = reward if done else reward + self.gamma * max(next_state_targets[i])
            i += 1

        self.train_network.fit(states, state_targets, epochs=1, verbose=0)
    
    
    def start(self):
        for _ in range(self.episodes):
            state = self.env.reset()
            
            while True:
                action = self.choose_action(state)

                next_state, reward, done = self.env.step(action)
                
                self.remember(state, action, next_state, reward, done)
                self.replay()                
                
                if done:
                    self.env.render()
                    break
                
                # Updates data at the end of the loop
                state = next_state         

            # Update epsilon for increase correct step probabilities
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

            # Save train network parameters
            if self.env.succeed():
                print('Something magic just happened')
                self.env.save(self.output_file)
                break

            self.target_network.set_weights(self.train_network.get_weights())