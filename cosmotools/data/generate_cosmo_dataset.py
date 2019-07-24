# This script is exectuted to create the histograms from the raw simulation file outputs...
# Currenty the path need to be set manually in the main function
# This script requires the package pynbody

# The used dataset was generated using the script Data/generate_data from commit bda27d7

import os, h5py, gc, pynbody
import numpy as np


def load_pynbody(filename_sim, path_sim):
    path_filename = os.path.join(path_sim, filename_sim)
    if not os.path.exists(path_filename):
        raise ValueError("PATH to simulation {} doesn't exist".format(path_filename))
    try:
        sim = pynbody.load(path_filename)
    except:
        print("File {} is not a pynbody simulation".format(path_filename))
        return None, -1

    print('   loadable_keys: {}'.format(sim.loadable_keys()))
    print('   properties: {}'.format(sim.properties))

    sim.physical_units()
    sim['pos'].convert_units('Mpc')

    lbox_size = sim.properties['boxsize'].in_units('Mpc')
    return sim, lbox_size


def generate_histogram(s, lbox, resolution):
    bin_nums = [resolution, resolution, resolution]
    H = np.histogramdd(s['pos'], bins=bin_nums, range=[[0, lbox], [0, lbox], [0, lbox]])[0]
    if np.isnan(np.sum(H)):
        raise ValueError('nan values found')
    H = np.array(H)
    return H

def main(mpc=350, resolution=266, nboxes=30):
    # TODO Non hardcoded params
    base_og_path = '/store/sdsc/sd01/comosology/data/nbody_raw_boxes/AndresBoxes/'
    path_og_data = base_og_path + 'Box_' + str(mpc) + 'Mpch_'
    path_new_data = '/store/sdsc/sd01/comosology/data/pre_processed_data/'

    if not os.path.exists(path_new_data):
        print('Creating path: {}'.format(path_new_data))
        os.makedirs(path_new_data)
    for i in range(nboxes):
        path_og_data_folder = path_og_data + str(i)
        print('Pasing folder: ' + path_og_data_folder)
        hist = np.zeros((resolution, resolution, resolution))
        for filename in sorted(os.listdir(path_og_data_folder)):
            if not filename.endswith(".txt") and not filename.endswith(".dat") and not filename.endswith(".info"):
                print(" * Loading file {}... ".format(filename))
                sim, lbox_size = load_pynbody(filename, path_og_data_folder)
                if sim:
                    print(" * Generate histogram for {}... ".format(filename))               
                    tmp_hist = generate_histogram(sim, lbox_size, resolution)
                    hist = np.add(hist, tmp_hist)
                del sim
                gc.collect()

        h5f = h5py.File(path_new_data + str(resolution) + '_' +"nbody_" + str(mpc) + "Mpc_" + str(i) + ".h5", 'w')
        h5f.create_dataset("data", data=hist)
        h5f.close()
    
    return 0


if __name__ == '__main__':
    main(mpc=350, resolution=266, nboxes=30)