import numpy as np
import matplotlib.pyplot as plt

data = np.array([1023.02, 981.27,  949.95,  931.92,  940.46,  931.92,
                 992.65,  1015.43, 947.10,  1136.90, 1132.16, 1122.67,
                 1030.61, 963.24,  922.43,  971.78,  1062.88, 943.31,
                 967.98,  918.63,  1003.09, 917.67,  939.51,  967.98,
                 1051.49, 1064.78, 985.06,  1016.38, 985.06,  958.49])

print(f"The sum is: {np.sum(data)}")
print(f"The mean is: {np.mean(data)}")
print(f"The variance is: {np.var(data)}")