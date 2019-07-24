if __name__ == '__main__':
	import sys, os
	sys.path.insert(0, os.path.join(os.path.dirname(__file__),'../../'))

import unittest

from cosmotools.testing import test_fmap
from cosmotools.testing import test_load_data

loader = unittest.TestLoader()

suites = []
suites.append(loader.loadTestsFromModule(test_fmap))
suites.append(loader.loadTestsFromModule(test_load_data))

suite = unittest.TestSuite(suites)


def run():  # pragma: no cover
    unittest.TextTestRunner(verbosity=2).run(suite)


if __name__ == '__main__':  # pragma: no cover
    os.environ["CUDA_VISIBLE_DEVICES"]=""

    run()