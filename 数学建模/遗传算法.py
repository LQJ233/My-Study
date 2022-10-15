import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

DNA_SIZE = 24
POP_SIZE = 200
CROSSOVER_RATE = 0.8#交叉概率
MUTATION_RATE = 0.005#突变率
N_GENERATIONS = 50#n代
X_BOUND = [-3, 3]#x边界
Y_BOUND = [-3, 3]


def F(x, y):#计算适应度
    return 3 * (1 - x) ** 2 * np.exp(-(x ** 2) - (y + 1) ** 2) - 10 * (x / 5 - x ** 3 - y ** 5) * np.exp(
        -x ** 2 - y ** 2) - 1 / 3 ** np.exp(-(x + 1) ** 2 - y ** 2)


def plot_3d(ax):#画图
    X = np.linspace(*X_BOUND, 100)#生成均匀步长的数字
    Y = np.linspace(*Y_BOUND, 100)
    X, Y = np.meshgrid(X, Y)#输入的x，y，就是网格点的横纵坐标列向量（非矩阵），输出的X，Y，就是坐标矩阵。坐标矩阵中x存储所有点的x坐标，y存储所有的y坐标
    Z = F(X, Y)
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm)
    ax.set_zlim(-10, 10)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.pause(3)
    plt.show()


def get_fitness(pop):#适应度方程，适应度高的留下，低的淘汰
    x, y = translateDNA(pop)
    pred = F(x, y)#适应度
    return (pred - np.min(
        pred)) + 1e-3  # 减去最小的适应度是为了防止适应度出现负数，通过这一步fitness的范围为[0, np.max(pred)-np.min(pred)],最后在加上一个很小的数防止出现为0的适应度


def translateDNA(pop):  # pop表示种群矩阵，一行表示一个二进制编码表示的DNA，矩阵的行数为种群数目
    #pop一行表示一个个体，每列就是特征
    x_pop = pop[:, 1::2]  # 奇数列表示X
    y_pop = pop[:, ::2]  # 偶数列表示y

    # pop:(POP_SIZE,DNA_SIZE)*(DNA_SIZE,1) --> (POP_SIZE,1)
    x = x_pop.dot(2 ** np.arange(DNA_SIZE)[::-1]) / float(2 ** DNA_SIZE - 1) * (X_BOUND[1] - X_BOUND[0]) + X_BOUND[0]
    y = y_pop.dot(2 ** np.arange(DNA_SIZE)[::-1]) / float(2 ** DNA_SIZE - 1) * (Y_BOUND[1] - Y_BOUND[0]) + Y_BOUND[0]
    return x, y
#dot()函数是矩阵乘,而*则表示逐个元素相乘
	#np.arange()函数返回一个有终点和起点的固定步长的排列
	#pop.dot(2 ** np.arange(DNA_SIZE)[::-1])已经转换成十进制
	#但是需要归一化到0~5,如有1111这么长的DNA,要产生的十进制数范围是[0, 15], 而所需范围是 [-1, 1],就将[0,15]缩放到[-1,1]这个范围
	#a[::-1]相当于 a[-1:-len(a)-1:-1]，也就是从最后一个元素到第一个元素复制一遍。所以你看到一个倒序
	#np.arange(DNA_SIZE)[::-1]得到10,9,8,...,0


def crossover_and_mutation(pop, CROSSOVER_RATE=0.8):
    new_pop = []
    for father in pop:  # 遍历种群中的每一个个体，将该个体作为父亲
        child = father  # 孩子先得到父亲的全部基因（这里我把一串二进制串的那些0，1称为基因）
        if np.random.rand() < CROSSOVER_RATE:  # 产生子代时不是必然发生交叉，而是以一定的概率发生交叉
            mother = pop[np.random.randint(POP_SIZE)]  # 再种群中选择另一个个体，并将该个体作为母亲
            cross_points = np.random.randint(low=0, high=DNA_SIZE * 2)  # 随机产生交叉的点
            child[cross_points:] = mother[cross_points:]  # 孩子得到位于交叉点后的母亲的基因
        mutation(child)  # 每个后代有一定的机率发生变异
        new_pop.append(child)

    return new_pop


#繁衍，有变异的基因会出现
def mutation(child, MUTATION_RATE=0.003):
    if np.random.rand() < MUTATION_RATE:  # 以MUTATION_RATE的概率进行变异
        mutate_point = np.random.randint(0, DNA_SIZE)  # 随机产生一个实数，代表要变异基因的位置
        child[mutate_point] = child[mutate_point] ^ 1  # 将变异点的二进制为反转


def select(pop, fitness):  # nature selection wrt pop's fitness
    # 适者生存的 select() 很简单, 我们只要按照适应程度 fitness 来选 pop 中的 parent 就好. fitness 越大, 越有可能被选到.
    idx = np.random.choice(np.arange(POP_SIZE), size=POP_SIZE, replace=True,
                           p=(fitness) / (fitness.sum()))
    return pop[idx]
#这里概率不能为负，所以pred要进行非负处理
	#replace表示抽样后是否放回，这里为True表示有放回，则可能会出现相同的索引值
    # p 就是选它的比例，按比例来选择适应度高的,也会保留一些适应度低的，因为也可能后面产生更好的变异
    #np.random.choice表示从序列中取值  np.arange()函数返回一个有终点和起点的固定步长的排列


def print_info(pop):
    fitness = get_fitness(pop)
    max_fitness_index = np.argmax(fitness)
    print("max_fitness:", fitness[max_fitness_index])
    x, y = translateDNA(pop)
    print("最优的基因型：", pop[max_fitness_index])
    print("(x, y):", (x[max_fitness_index], y[max_fitness_index]))


if __name__ == "__main__":
    fig = plt.figure()#创建绘图对象，
    ax = Axes3D(fig)#用这个绘图对象创建Axes对象。
    plt.ion()  # 将画图模式改为交互模式，程序遇到plt.show不会暂停，而是继续执行
    plot_3d(ax)

    pop = np.random.randint(2, size=(POP_SIZE, DNA_SIZE * 2))  # matrix (POP_SIZE, DNA_SIZE)随机生成一个0到2的数
    # pop表示种群矩阵，一行表示一个二进制编码表示的DNA，矩阵的行数为种群数目
    print('pop=',pop)
    for _ in range(N_GENERATIONS):  # 迭代N代
        x, y = translateDNA(pop)
        if 'sca' in locals():
            sca.remove()
        sca = ax.scatter(x, y, F(x, y), c='black', marker='o');
        plt.show();
        plt.pause(0.1)
        pop = np.array(crossover_and_mutation(pop, CROSSOVER_RATE))
        # F_values = F(translateDNA(pop)[0], translateDNA(pop)[1])#x, y --> Z matrix
        fitness = get_fitness(pop)
        pop = select(pop, fitness)  # 选择生成新的种群

    print_info(pop)
    plt.ioff()
    plot_3d(ax)