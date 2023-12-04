import setuptools

setuptools.setup(
    name='CrysFieldExplorer',
    version='1.0.0',    
    description='Fast optimization of CEF Hamiltonian',
    url='https://github.com/KyleQianliMa/CrysFieldExplorer',
    author='Kye Ma',
    author_email='maq1@ornl.gov',
    license='MIT License',
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    install_requires=['matplotlib',
                      'scipy',
                      'cma', 'numpy' ,
                      'pyswarm','mpi4py'
                      'alive_progress',
                      'pdb',
                      'sympy',
                      'scipy',                      
                      ],
)
