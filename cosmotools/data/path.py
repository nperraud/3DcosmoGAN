import socket
import os

def root_path():
    ''' Defining the different root path using the host name '''
    hostname = socket.gethostname()
    # Check if we are on pizdaint
    if 'nid' in hostname:
        rootpath = '/scratch/snx3000/nperraud/pre_processed_data/' 
    elif 'omenx' in hostname:
        rootpath = '/store/nati/datasets/cosmology/pre_processed_data/'         
    else:
        # This should be done in a different way
        utils_module_path = os.path.dirname(__file__)
        rootpath = utils_module_path + '/../../data/nbody/preprocessed_data/'
    return rootpath