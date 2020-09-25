from setuptools import setup, find_packages

setup(
    name                = 'nutellaAgent',
    version             = '0.1',
    description         = 'for nutella service',
    author              = 'songpie',
    author_email        = 'songmi.ohh@gmail.com',
    license             = 'MIT'
    install_requires    =  [],
    packages            = find_packages(exclude = []),
    keywords            = ['nutellaAgent'],
    python_requires     = '>=3',
    package_data        = {},
    zip_safe            = False,
    classifiers         = [
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.2',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
)