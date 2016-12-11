from distutils.core import setup
from glob import glob
scripts = glob('bin/*')

setup(name='SloppyJoes',
        version='0.1',
        description='Efficient implementation of Geodesic acceleration for sloppy models.',
        authors=['Sean McLaughlin','Mitchell McIntire'],
        author_email='swmclau2@stanford.edu',
        url='https://github.com/mclaughlin6464/SloppyJoes',
        scripts=scripts,
        packages=['SloppyJoes', 'geodesicLM'])
