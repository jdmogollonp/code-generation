from itertools import chain, zip_longest
from random import sample, choices, seed
import os
import numpy as np

class Generator(object):
    def __init__(self, training_dir, testing_dir):
        seed(0)
        self.training_dir = training_dir
        self.testing_dir = testing_dir


    def function_generator(self):
        """
        Generate a random function which return an arithmetic function using [+, -, *] operators.
        The function can have one to three parameters.
        """
        # letras
        letras = [*choices('a', k=sample(range(1,3),k=1)[0]) ,*choices('b', k=sample(range(0,3),k=1)[0]),*choices('c', k=sample(range(0,3),k=1)[0])]

        if letras=='':
            letras = [*choices('a', k=sample(range(1,3),k=1)[0]) ,*choices('b', k=sample(range(1,3),k=1)[0]),*choices('c', k=sample(range(1,3),k=1)[0])]

        # numeros
        numeros = sample(range(1,100),  k= len(letras))
        numeros = [str(i) for i in numeros]
        letras_numeros = [x for x in chain(*zip_longest(letras, numeros)) if x is not None]

        #operacion
        operador = ['+', '-', '*']
        operadores = choices(operador, k=(len(letras_numeros)-1))
        operacion = [x for x in chain(*zip_longest(letras_numeros, operadores)) if x is not None]
        operacion = ' '.join(operacion)

        # declaracion
        distintos_letras = list(set(letras))
        declaracion = [x for x in chain(*zip_longest(distintos_letras, [',' for i in range( 0,len(distintos_letras)-1 )])) if x is not None]
        declaracion = ''.join(declaracion)
        python_funcion = f'def f({declaracion}):\n\treturn {operacion}\n# <end>'
        return python_funcion


    def generate_training_set(self, filename, size):
        """
        Generate a file that contains {size} different functions generated with function_generator and store it in the 
        {training_dir} with name {filename}.
        """
        self.training_set = set()
        while len(self.training_set) < size:
            self.training_set.add(self.function_generator())

        with open(os.path.join(self.training_dir, f'{filename}.py'), 'w') as f:
            f.write('\n\n'.join(self.training_set))


    def generate_test_set(self, file_prefix, size, cases_set_size, different=True, sep=','):
        if different and self.training_set is not None:
            self.testing_set = set()
            while len(self.testing_set) < size:
                fun = self.function_generator()
                if fun in self.training_set:
                    continue
                self.testing_set.add(fun)
        else:
            self.testing_set = set()
            while len(self.testing_set) < size:
                self.testing_set.add(self.function_generator())

        for idx, fun in enumerate(self.testing_set):
            with open(os.path.join(self.testing_dir, f'{file_prefix}_{idx}.py'), 'w') as f:
                f.write(fun)
            
            self.generate_test_case(fun, f'{file_prefix}_{idx}', cases_set_size, sep)

    
    def generate_test_case(self, fun, file_prefix, size, sep):
        """
        Generate {size} test cases for the specific function and store them in a csv called {file_prefix}.csv in the folder {self.testing_dir}
        """
        variables = fun.split('\n')[0][6:-2].split(',') # Get the name of the variables
        ops = fun.split('\n')[1][8:] # Get the operations
        variables_data = np.random.choice(5 * size * len(variables), (size, len(variables)), replace=False) # Generate random values for the variable

        # Evaluate the function to get the values
        data = np.zeros((size, len(variables) + 1), dtype=int)
        for i, case in enumerate(variables_data):
            vars_dict = {}
            for j, val in enumerate(case):
                data[i, j] = val
                vars_dict[variables[j]] = val
            data[i, -1] = eval(ops, vars_dict)

        np.savetxt(os.path.join(self.testing_dir, f'{file_prefix}.csv'), data, delimiter=sep, fmt='%i')   
    