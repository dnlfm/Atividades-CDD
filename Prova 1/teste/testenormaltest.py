import pandas as pd
import numpy as np
import scipy.stats as stats
from scipy.stats import normaltest

x = stats.norm.rvs(size = 20000)

print(normaltest(x))

y = stats.uniform.rvs(size = 100)

print(normaltest(y))
