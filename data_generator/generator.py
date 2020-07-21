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
        python_funcion = """def f("""+ declaracion +"""):\n\t return """+operacion+ """ #<end>"""
        return(python_funcion)
    s
    def generate_training_set(self,size):
        self.training_set = {self.function_generator():1 for i in range(0,size)}
        self.list_training_set = list(self.training_set.keys())
        return self.list_training_set

    def generate_test_set(self,path,set_size,cases_set_size):
        self.path = path
        test_set = {self.function_generator():1 for i in range(0,set_size)}
        self.test_set = {key:1  for key in test_set.keys() if key not in  self.training_set.keys() }
        list_keys = list(self.test_set.keys())
        self.test_cases = { 'test_cases ' + j :  pd.concat([self.generate_test_case(j) for i in range(0,cases_set_size)])  for  j in   list_keys}
        for i, key in enumerate(x.test_cases):
            self.export_file( self.path,'.csv','test_cases_' + str(i), self.test_cases[key] )
        self.list_testing_set = list(self.test_set.keys())
        return self.list_testing_set


    def generate_test_case(self,function_str):
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
    
    def export_file(self, path, extension, file_name,data):
        if extension == '.csv':
            return data.to_csv(str(file_name)+ '.csv', sep=',' , index=False, header=False)
        elif extension == '.py':
            file_name = path+file_name+'.py'
            with open(file_name, 'w') as f:
                f.write(data)        
            return str(file_name)+ ' exported' ''                
    