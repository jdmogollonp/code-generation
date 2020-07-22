from data_generator.generator import Generator
from pretrainer.pretrainer import Pretrainer
from env.env import Environment
from drqn_agent.ddrqn_agent import DDRQNAgent

# Generate data
#generator = Generator('./data/training/', './data/testing/')
#generator.generate_training_set('training_set', 1000)
#generator.generate_test_set('testing_example', 10, 100)

# Pre-train the model
pretrainer = Pretrainer('./data/training/', '/Users/miller/Desktop/training_checkpoints')
pretrainer.prepare_data()
# pretrainer.train()

# Train the model for specific algorithm
env = Environment('def f(a):', pretrainer.idx2char, pretrainer.char2idx, 'f', 1, './data/testing/testing_example_0.csv', pretrainer.seq_length)
agent = DDRQNAgent(len(pretrainer.vocab), env, pretrainer.checkpoint_dir, 
                    pretrainer.embedding_dim, pretrainer.rnn_units, 'output/result.py')
agent.start() 