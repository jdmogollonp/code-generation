import ast
import pandas as pd

class Tester(object):
    def __init__(self, func_name, num_params, test_cases_csv, csv_sep=',', header=None, error_ignore=True):
        self.error_ignore = error_ignore
        self.func_name = func_name
        self.num_params = num_params
        self.test_cases = pd.read_csv(test_cases_csv, sep=csv_sep, header=header)
    

    def is_valid(self, code):
        try:
            ast.parse(code)
            self.test(code, once=True)
        except:
            return False
        return True


    def test(self, code, once=False):
        exec(code)
        total, passed = 0., 0.
        first = True
        for _, row in self.test_cases.iterrows():
            if once and not first:
                break
            if len(row) != self.num_params + 1:
                if self.error_ignore:
                    continue
                raise Exception('Number of parameters is not as expected')
            
            s = f'{self.func_name}('
            for i in range(0, len(self.test_cases.columns) - 1):
                if type(row[self.test_cases.columns[i]]) == str:
                    s += eval(row[self.test_cases.columns[i]]) + ', '
                else:
                    s += str(row[self.test_cases.columns[i]]) + ', '
            s = s[:-2] + ')'
            if eval(s) == row[self.test_cases.columns[-1]]:
                passed += 1.
            total += 1.

            if first:
                first = False
        return passed/total