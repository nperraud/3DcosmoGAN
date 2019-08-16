# Module to download the dataset.

import os

import sys
if sys.version_info[0] > 2:
    from urllib.request import urlretrieve
else:
    from urllib import urlretrieve
import hashlib
import zipfile


def require_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return None


def download(url, target_dir, filename=None):
    require_dir(target_dir)
    if filename is None:
        filename = url_filename(url)
    filepath = os.path.join(target_dir, filename)
    urlretrieve(url, filepath)
    return filepath


def url_filename(url):
    return url.split('/')[-1].split('#')[0].split('?')[0]


def check_md5(file_name, orginal_md5):
    # Open,close, read file and calculate MD5 on its contents
    with open(file_name, 'rb') as f:
        hasher = hashlib.md5()  # Make empty hasher to update piecemeal
        while True:
            block = f.read(64 * (
                1 << 20))  # Read 64 MB at a time; big, but not memory busting
            if not block:  # Reached EOF
                break
            hasher.update(block)  # Update with new block
    md5_returned = hasher.hexdigest()
    # Finally compare original MD5 with freshly calculated
    if orginal_md5 == md5_returned:
        print('MD5 verified.')
        return True
    else:
        print('MD5 verification failed!')
        return False


def unzip(file, targetdir):
    with zipfile.ZipFile(file, "r") as zip_ref:
        zip_ref.extractall(targetdir)


if __name__ == '__main__':
    # The dataset is availlable at https://doi.org/10.5281/zenodo.1303272
    
    url_nbody = 'https://zenodo.org/record/3369574/files/nbody-cubes.zip?download=1'
    url_readme = 'https://zenodo.org/record/3369574/files/README.md?download=1'

    md5_nbody = 'abc89d98e60d94fda703f5d176594dd9'
    md5_readme = '0d88b246880d554406e48d249931008d'

    print('Download README')
    download(url_readme, 'data/nbody')
    assert (check_md5('data/nbody/README.md', md5_readme))

    print('Download nbody-cubes')
    download(url_nbody, 'data/nbody/preprocessed_data')
    assert(check_md5('data/nbody/preprocessed_data/nbody-cubes.zip', md5_nbody))
    print('Extract nbody-cubes')
    unzip('data/nbody/preprocessed_data/nbody-cubes.zip', 'data/nbody/')
    os.remove('data/nbody/preprocessed_data/nbody-cubes.zip')


