#本功能实现最小值的求解#

from matplotlib import pyplot as plt
import numpy as np
import random
import math
plt.ion()#这里需要把matplotlib改为交互状态

#初始值设定
hi=3#x的最大值，要根据题目和具体函数来定
lo=-3#x的最小值
alf=0.95#退火率
T=100#初始温度

#目标函数
def f(x):
    return 11*np.sin(x)+7*np.cos(5*x)##注意这里要是np.sin，#初始函数，求解这个函数的最小值

#可视化函数（开始清楚一次然后重复的画）
def visual(x):
    plt.cla()
    plt.axis([lo-1,hi+1,-20,20])
    m=np.arange(lo,hi,0.0001)
    plt.plot(m,f(m))
    plt.plot(x,f(x),marker='o',color='black',markersize='4')
    plt.title('temperature={}'.format(T))
    plt.pause(0.1)#如果不停啥都看不见

#随机产生初始值
def init():
    return random.uniform(lo,hi)#在函数定义域内随机生成一个数，作为初始的x，

#新解的随机产生
def new(x):
    x1=x+T*random.uniform(-1,1)
    if (x1<=hi)&(x1>=lo):#如果x1这个新解在定义域内，就返回x1
        return x1
    elif x1<lo:#如果新解小于下限
        rand=random.uniform(-1,1)
        return rand*lo+(1-rand)*x
    else:#如果新解大于上限
        rand=random.uniform(-1,1)
        return rand*hi+(1-rand)*x

#p函数
def p(x,x1):
    return math.exp(-abs(f(x)-f(x1))/T)#小概率接受更差的结果

def main():
    global x
    global T
    x=init()
    while T>0.0001:#eps=0.0001
        visual(x)
        for i in range(500):
            x1=new(x)#x1为新的数
            if f(x1)<=f(x):#如果更优
                x=x1#取代以前的值
            else:
                if random.random()<=p(x,x1):#否则小概率接受更差的值，目的是为了跳出局部最优解
                    x=x1
                else:
                    continue
        T=T*alf#退火一次
    print('最小值为：{}'.format(f(x)))

main()