if __name__ == '__main__':
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

import unittest

from cosmotools.data import load, fmap
import numpy as np
from gantools.blocks import np_downsample_2d, np_downsample_3d, np_downsample_1d

class TestDataLoad(unittest.TestCase):
    def test_cosmo(self):
        forward = fmap.forward
        dataset = load.load_nbody_dataset(
            ncubes=None, spix=32, Mpch=350, forward_map=forward, patch=True)
        it = dataset.iter(10)
        print(next(it).shape)
        assert (next(it).shape == (10, 32, 32, 4))
        del it, dataset

        dataset = load.load_nbody_dataset(
            ncubes=None,
            spix=32,
            Mpch=350,
            forward_map=forward,
            patch=True,
            is_3d=True)
        it = dataset.iter(4)
        print(next(it).shape)
        assert (next(it).shape == (4, 32, 32, 32, 8))
        del it, dataset

        dataset = load.load_nbody_dataset(
            ncubes=None, spix=32, Mpch=70, forward_map=None, patch=False)
        it = dataset.iter(10)
        print(next(it).shape)

        assert (next(it).shape == (10, 32, 32, 1))
        del it, dataset

        dataset = load.load_nbody_dataset(
            ncubes=2, spix=256, Mpch=70, forward_map=forward, patch=False)
        assert (dataset.get_all_data().shape[0] == 256 * 2)
        del dataset

        dataset = load.load_nbody_dataset(
            ncubes=2, spix=128, Mpch=350, forward_map=forward, patch=False)
        it = dataset.iter(10)
        print(next(it).shape)
        assert (next(it).shape == (10, 128, 128, 1))
        del it, dataset
        dataset1 = load.load_nbody_dataset(
            ncubes=4, spix=128, Mpch=350, forward_map=forward, patch=False, shuffle=False, augmentation=False, scaling=2, is_3d=True)
        it1 = dataset1.iter(3)
        s1 = next(it1)
        del it1, dataset1

        dataset2 = load.load_nbody_dataset(
            ncubes=4, spix=32, Mpch=350, forward_map=forward, patch=False, shuffle=False, augmentation=False, scaling=8, is_3d=True)
        it2 = dataset2.iter(3)
        s2 = next(it2)
        del it2, dataset2
        np.testing.assert_allclose(np_downsample_3d(s1,4), s2)

        dataset1 = load.load_nbody_dataset(
            ncubes=2, spix=128, Mpch=350, forward_map=forward, patch=False, shuffle=False, augmentation=False, scaling=2)
        it1 = dataset1.iter(10)
        s1 = next(it1)
        del it1, dataset1

        dataset2 = load.load_nbody_dataset(
            ncubes=2, spix=32, Mpch=350, forward_map=forward, patch=False, shuffle=False, augmentation=False, scaling=8)
        it2 = dataset2.iter(10)
        s2 = next(it2)
        del it2, dataset2
        np.testing.assert_allclose(np_downsample_2d(s1,4), s2)


if __name__ == '__main__':
    unittest.main()