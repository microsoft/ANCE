from setuptools import setup

with open('SEED-Encoder.md') as f:
    readme = f.read()

setup(
   name='SEED-Encoder',
   long_description=readme,
   install_requires=[
        'scikit-learn',
        'pandas',
        'tensorboardX',
        'tqdm',
        'tokenizers==0.9.2',
        'six',
    ],
)