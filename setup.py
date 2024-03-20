from setuptools import setup, find_packages

setup(
    name='fewshot',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'fire',
        'ir-measures',
        'ir_datasets',
        'pandas',
        'pyterrier',
        'transformers',
        'torch',
        'tqdm',
        'more-itertools'
    ],
    entry_points={
        'console_scripts': [
            'run_topics = fewshot.evaluate.run_topics:run',
            'run_metrics = fewshot.evaluate.run_metrics:main',
        ],
    },
)