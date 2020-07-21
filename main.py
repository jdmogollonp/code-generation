from data_generator.generator import Generator
from pretrainer.pretrainer import Pretrainer
from env.env import Environment
from drqn_agent.ddrqn_agent import DDRQNAgent

# Generate data
generator = Generator()
# TODO: ...

# Pre-train the model
pretrainer = Pretrainer('./data', '/Users/miller/Desktop/training_checkpoints')
pretrainer.train()

# Train the model for specific algorithm
env = Environment('def f(a)', pretrainer.idx2char, 'f', 1, 'test_cases_1.csv', pretrainer.seq_length)
agent = DDRQNAgent(len(pretrainer.vocab), env, pretrainer.checkpoint_dir, 
                    pretrainer.embedding_dim, pretrainer.rnn_units, 'output/result.py')
agent.start()