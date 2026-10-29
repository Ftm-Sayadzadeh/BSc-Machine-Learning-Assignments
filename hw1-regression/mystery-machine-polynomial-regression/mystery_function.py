# import pandas as pd
# import numpy as np
import math

def ceil_to_precision(value, precision):
    factor = 10 ** precision
    return math.ceil(value * factor) / factor

# df = pd.read_excel('train.xlsx')

# x = df['x'].values
# y = df['y'].values
# z = df['z'].values
# f_target = df['F(x, y, z)'].values

# X_poly = np.column_stack([
#     np.ones(x.shape[0]), # Bias (w0)
#     x, x**2, x**3,
#     y, y**2, y**3,
#     z, z**2, z**3,
#     x*y, x**2*y, x*y**2,
#     x*z, x**2*z, x*z**2,
#     y*z, y**2*z, y*z**2,
#     x*y*z
# ])

# # (X^T * X)w = X^T * f
# xtx = X_poly.T @ X_poly
# xty = X_poly.T @ f_target
# w = np.linalg.solve(xtx, xty)

# print("Calculated Weights (w_0 to w_19):")
# for i, coef in enumerate(w):
#     print(f"w_{i} = {coef:.4f}")


# def predict_poly(x_val, y_val, z_val, weights):
#     features = np.array([
#         1,
#         x_val, x_val**2, x_val**3,
#         y_val, y_val**2, y_val**3,
#         z_val, z_val**2, z_val**3,
#         x_val*y_val, x_val**2*y_val, x_val*y_val**2,
#         x_val*z_val, x_val**2*z_val, x_val*z_val**2,
#         y_val*z_val, y_val**2*z_val, y_val*z_val**2,
#         x_val*y_val*z_val
#     ])
#     return np.dot(features, weights)


def F(x, y, z):
    b = -0.0000
    w = [0.5000 ,1.0000 ,0.2500,
     0.6000 ,0.2000 ,0.4000, 0.2500,
     0.1100 ,0.2300,0.3100 ,0.7600,
     0.1400 ,0.1500 ,0.6200,0.8900,
     0.3000 ,0.1000, 0.6000 ,0.2100]
    return (b + w[0] * x + w[1] * x ** 2 + w[2] * x ** 3 +
            w[3] * y + w[4] * y ** 2 + w[5] * y ** 3 +
            w[6] * z + w[7] * z ** 2 + w[8] * z ** 3 +
            w[9] * x * y + w[10] * x ** 2 * y + w[11] * x * y ** 2 +
            w[12] * x * z + w[13] * x ** 2 * z + w[14] * x * z ** 2 +
            w[15] * y * z + w[16] * y ** 2 * z + w[17] * y * z ** 2 +
            w[18] * x * y * z)


x = float(input())
y = float(input())
z = float(input())
print(round(F(x, y, z), 2))