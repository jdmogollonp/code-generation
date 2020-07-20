from itertools import combinations,chain,zip_longest
from random import sample, randrange, choices, seed
from itertools import chain
import string
import re
import random
import pandas as pd

class Generator(object):
    def __init__(self):
        seed(90)
    

    def function_generator(self):
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

        python_funcion = """def f("""+ declaracion +"""):\n\treturn """+operacion+ """ #<end>"""
        return(python_funcion)
    

    def generate_training_set(self, size):
        self.training_set = {self.function_generator():1 for i in range(0,size)}
        return list(self.training_set.keys())
    

    def generate_test_set(self, size):
        test_set = {self.function_generator():1 for i in range(0,size)}
        self.test_set = {key:1  for key in test_set.keys() if key not in  self.training_set.keys() }
        return list(self.test_set.keys())
    

    def generate_test_case(self, function_str):
        function = (function_str.split('def'))[1].split(':')[0]
        operation = (function_str.split('return'))[1].split('#')[0]
        variables = re.findall(r'\(([^()]+)\)', function)
        variables = ' '.join(variables)
        variables = variables.split(",")
        variables_dict = { i:random.randint(0,100) for i in  variables}
        df_result = pd.DataFrame([variables_dict])
        df_result['resultado'] = eval(operation, variables_dict)
        df_result.sort_index(axis=1, inplace = True)
        df_result.reset_index(drop=True, inplace = True)
        return df_result
    

    def cases_to_csv(self, function_str, size):
        cases_list =[self.generate_test_case(function_str) for i in range(0,size)] 
        cases_list = pd.concat(cases_list)
        cases_list.reset_index(drop=True, inplace= True)
        self.cases_list = cases_list
        return cases_list.to_csv('cases_list.csv',   sep=',',index=False, header=False)
        