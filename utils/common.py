import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
sys.path.insert(0,currentdir)


from include import *
from draw import *
from file import *

from net.pytorchviz import *
from net.callback.rate import *

from net.layer.function import *
from net.layer.sync_batchnorm import *


# kaggle ---
# DATA_DIR = '/root/share/project/kaggle/google_doodle/data'
# DATA_DIR = '/dataset/doodle-quickdraw'
# DATA_DIR = '../data/train_key_id_CV/all/'
DATA_DIR = '../data/train_key_id_CV/all_80/'




#---------------------------------------------------------------------------------

class Struct(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)



#---------------------------------------------------------------------------------
print('@%s:  ' % os.path.basename(__file__))

if 1:
    SEED = 35202  #123  #int(time.time()) #
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    print ('\tset random seed')
    print ('\t\tSEED=%d'%SEED)

if 1:
    torch.backends.cudnn.benchmark = True  ##uses the inbuilt cudnn auto-tuner to find the fastest convolution algorithms. -
    torch.backends.cudnn.enabled   = True
    print ('\tset cuda environment')
    print ('\t\ttorch.__version__              =', torch.__version__)
    print ('\t\ttorch.version.cuda             =', torch.version.cuda)
    print ('\t\ttorch.backends.cudnn.version() =', torch.backends.cudnn.version())
    try:
        print ('\t\tos[\'CUDA_VISIBLE_DEVICES\']     =',os.environ['CUDA_VISIBLE_DEVICES'])
        NUM_CUDA_DEVICES = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    except Exception:
        print ('\t\tos[\'CUDA_VISIBLE_DEVICES\']     =','None')
        NUM_CUDA_DEVICES = 1

    print ('\t\ttorch.cuda.device_count()      =', torch.cuda.device_count())
    #print ('\t\ttorch.cuda.current_device()    =', torch.cuda.current_device())


print('')

#---------------------------------------------------------------------------------
## useful : http://forums.fast.ai/t/model-visualization/12365/2
