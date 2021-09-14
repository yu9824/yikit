
'''
Copyright (c) 2021 yu9824

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''

import sys

DATETIME_KEY = '%Y%m%d-%H%M-%S'
DATE_KEY = '%Y%m%d'

def is_notebook():
    """the environment is notebook or not.
    reference: https://blog.amedama.jp/entry/detect-jupyter-env

    Returns
    -------
    bool
        is_notebook or not.
    """
    if 'get_ipython' not in globals():
        return 'ipykernel' in sys.modules
        # Python shell
        # return False
    env_name = get_ipython().__class__.__name__
    if env_name == 'TerminalInteractiveShell':
        # IPython shell
        return False
    return True

'''
例
if 'ipykernel' in sys.modules:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm
'''

if __name__ == '__main__':
    pass