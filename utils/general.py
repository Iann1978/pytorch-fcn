import glob
import os
from pathlib import Path
def check_file(file):
    # Searches for file if not found locally
    if os.path.isfile(file):
        return file
    else:
        #files = glob.glob('./**/' + file, recursive=True)  # find file
        #assert len(files), 'File Not Found: %s' % file  # assert file was found
        #return files[0]  # return first file if multiple found
        assert None, 'File Not Found: %s' % file #assert file was found
def increment_dir(dir, comment=''):
    # Increments a directory runs/exp1 --> runs/exp2_comment
    n = 0  # number
    dir = str(Path(dir))  # os-agnostic
    d = sorted(glob.glob(dir + '*'))  # directories
    if len(d):
        n = max([int(x[len(dir):x.find('_') if '_' in x else None]) for x in d]) + 1  # increment
    return dir + str(n) + ('_' + comment if comment else '')
