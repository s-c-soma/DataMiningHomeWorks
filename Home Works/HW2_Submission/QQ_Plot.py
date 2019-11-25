import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.gofplots import qqplot_2samples

x_secA = np.array([47, 63, 71, 39, 47, 49, 43, 37, 81, 69, 38, 13, 29, 61, 49, 53, 57, 23, 58, 17, 73, 33, 29])
y_secB = np.array([20, 49, 85, 17, 33, 62, 93, 64, 37, 81, 22, 18, 45, 42, 14, 39, 67, 47, 53, 73, 58, 84, 21])
print('Sec A:', np.sort(x_secA))
print('Sec B:', np.sort(y_secB))
print("A sec: 25th percentile: ", np.percentile(x_secA, 25))
print("A sec: Median: ", np.median(x_secA))
print("A sec: 75th percentile: ", np.percentile(x_secA, 75))

print("B sec: 25th percentile: ", np.percentile(y_secB, 25))
print("B sec: Median: ", np.median(y_secB))
print("B sec: 75th percentile: ", np.percentile(y_secB, 75))
pp_x = sm.ProbPlot(x_secA)
pp_y = sm.ProbPlot(y_secB)
qqplot_2samples(pp_x, pp_y)


# compare x quantiles to y quantiles
fig3 = pp_x.qqplot(other=pp_y, line='q', xlabel='Section B', ylabel='Section A')
plt.show()

#section B is doing better