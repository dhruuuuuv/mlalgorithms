# machine learning algorithms
a selection of machine learning algorithms implemented in Python

DETAILS
---

The programs are written using python3, so assuming that python3 is installed on your machine, the whole program can be run by executing:

    python main.py

after unzipping all the files.

DEPENDENCIES
---

The python programs use the following dependencies:

    import csv
    import math
    import operator

    import matplotlib.pyplot as plt
    import numpy as np

where matplotlib and numpy may need to be installed seperately. This can be done using pip.

FILE STRUCTURE
---

The main.py file gives an example of how to use the algorithms. The functions exist in seperate modules and can be seen from the import section of the main file:

    import KNN
    import crossval
    import datanorm
    import linearreg

which clearly refer to the K-Nearest Neighbour, Cross-Validation, Data-Normalisation and the Linear Regression Algorithm implementations. They may also include other helper functions.
