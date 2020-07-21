from data_generator.generator import Generator
from pretrainer.pretrainer import Pretrainer
from env.env import Environment
from drqn_agent.ddrqn_agent import DDRQNAgent

# Generate data
generator = Generator()
# TODO: ...

generator.generate_training_set(1000)
generator.generate_test_set('C:/Users/Juan Mogollon/Desktop/ML/Proyecto/',10,100)

for i in range(0,len(generator.list_testing_set)):
    generator.export_file('C:/Users/Juan Mogollon/Desktop/ML/Proyecto/','.py','testing_function_'+ str(i),generator.list_testing_set[i])

generator.export_file('C:/Users/Juan Mogollon/Desktop/ML/Proyecto/','.py','training_set',  '\n'.join( generator.list_training_set) )

# Pre-train the model
pretrainer = Pretrainer('./data', '/Users/miller/Desktop/training_checkpoints')
pretrainer.train()

# Train the model for specific algorithm
env = Environment('def f(a)', pretrainer.idx2char, 'f', 1, 'test_cases_1.csv', pretrainer.seq_length)
agent = DDRQNAgent(len(pretrainer.vocab), env, pretrainer.checkpoint_dir, 
                    pretrainer.embedding_dim, pretrainer.rnn_units, 'output/result.py')
agent.start() 