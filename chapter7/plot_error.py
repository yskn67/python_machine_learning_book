# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
from pmf import ensemble_error


error_range = np.arange(0.0, 1.01, 0.01)
ens_errors = [ensemble_error(11, error)
              for error in error_range]

plt.plot(error_range, ens_errors,
         label='Ensemble error',
         linewidth=2)
plt.plot(error_range, error_range,
         linestyle='--', label='Base error',
         linewidth=2)
plt.xlabel('Base error')
plt.ylabel('Base/Ensemble error')
plt.legend(loc='upper left')
plt.grid()
plt.show()
