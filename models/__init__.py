import os

path = os.getcwd()
opj = os.path.join
inter_path = 'models' if 'models' not in path else ''
model_path = opj('.', inter_path)
try:
    __all__ = [m.split('.py')[0] for m in os.listdir('./' + inter_path) if
               m not in ['__init__.py', '__pycache__'] and not os.path.isdir(opj(model_path, m))]  # only works when running code on the upper directory

    for model in __all__:
        exec(f'from models import {model}')
except:
    print('not running in expected path')

