import os.path as osp
import sys

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

this_dir = osp.dirname(__file__)

package_path = osp.join(this_dir, '.', 'net')
add_path(package_path)

package_path = osp.join(this_dir, '.', 'net', 'customed_net')
add_path(package_path)

package_path = osp.join(this_dir, '.', 'preprocess')
add_path(package_path)

package_path = osp.join(this_dir, '.', 'utils')
add_path(package_path)

