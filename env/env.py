from .tester import Tester

class Environment(object):
    def __init__(self, initial_state, int2char, char2int, func_name, num_params, test_cases_csv, 
                max_seq_length, csv_sep=',', header=None, error_ignore=True):
        self.initial_state = [char2int[i] for i in initial_state]
        self.state = self.initial_state.copy()
        self.tester = Tester(func_name, num_params, test_cases_csv, csv_sep, header, error_ignore)
        self.int2char = int2char
        self.max_seq_length = max_seq_length


    def reset(self):
        self.state = self.initial_state.copy()
        return self.state


    def done(self):
        if len(self.state) < 5:
            return False
        return len(self.state) > self.max_seq_length or ''.join([self.int2char[i] for i in self.state[-5:]]) == '<end>'


    def state_to_code(self):
        return ''.join([self.int2char[i] for i in self.state])


    def reward(self):
        if not self.done() or not self.tester.is_valid(self.state_to_code()):
            return -1
        print('To test:')
        print(self.state_to_code())
        return 10 * self.tester.test(self.state_to_code())


    def step(self, action):
        self.state.append(action)
        return self.state, self.reward(), self.done()
    

    def succeed(self):
        return self.tester.is_valid(self.state_to_code()) and (self.tester.test(self.state_to_code()) > 0.2)
    

    def render(self):
        print(self.state_to_code())


    def save(self, output_path):
        with open(output_path, 'w') as f:
            f.write(self.state_to_code())
