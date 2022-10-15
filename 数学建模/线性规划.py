"""标准形式为：
min z=2X1+3X2+x
s.t
x1+4x2+2x3>=8
3x1+2x2>=6
x1,x2,x3>=0

上述线性规划问题Python代码
"""
# 导入相关库
import numpy as np
from scipy import optimize as op

# 定义决策变量范围
x1 = (0, None)
x2 = (0, None)
x3 = (0, None)

# 定义目标函数系数
c = np.array([2, 3, 1])

# 定义约束条件系数
A_ub = np.array([[-1, -4, -2], [-3, -2, 0]])
B_ub = np.array([-8, -6])

# 求解
res = op.linprog(c, A_ub, B_ub, bounds=(x1, x2, x3))
print(res)