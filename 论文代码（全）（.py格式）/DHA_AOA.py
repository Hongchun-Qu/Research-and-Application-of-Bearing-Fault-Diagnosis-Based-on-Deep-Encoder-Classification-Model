import numpy as np
import random
import matplotlib.pyplot as plt
import math
import shujuchuli
from DHA_CDA import DHA_CDA
from DHA_SDA import DHA_SDA
from DHA_DDA import DHA_DDA
import heapq
from sklearn.decomposition import PCA
import matplotlib
import copy
import os



'''
论文：The Arithmetic Optimization Algorithm
该算法是一种基于群体的元启发式算法，能够在不计算其导数的情况下解决优化问题。
管元启发式算法在基于群体的优化方法领域存在差异，但优化过程包括两个主要阶段:探索 exploration和开发exploitation。
前者是指使用算法的搜索代理广泛覆盖搜索空间，以避免局部解。后者是勘探阶段所得解的精度提高。
AOA提议的探索(多样化)和开发(强化)机制，通过数学中的算术运算符实现的(即1)乘法(M″×)、(2)除法(D″)、(3)减法(S″)和(4)加法(A″+))实现
乘除用于exploration阶段，加减用于exploitation阶段
首先随机生成一组候选解；
数学优化器加速（Math Optimizer Accelerated function）函数：MOA
位置更新方程
'''



def fit_fun(X, data, data_label, encoder, train_step, fault_num):  # 适应函数

    if encoder == 'sda_para':
        sda = DHA_SDA(X[0], X[1], X[2], lr=X[3], train_step=train_step, fault_num=fault_num)
        _, _, _, _, _, cost, accuracy = sda.fit(data, data_label, -2)
    elif encoder == 'cda_para':
        cda = DHA_CDA(X[0], X[1], X[2], lr=X[3], train_step=train_step, fault_num=fault_num)
        _, _, _, _, _, cost, accuracy = cda.fit(data, data_label, -2)
    elif encoder == 'dda_para':
        dda = DHA_DDA(X[0], X[1], X[2], lr=X[3], train_step=train_step, fault_num=fault_num)
        _, _, _, _, _, cost, accuracy = dda.fit(data, data_label, -2)

    return cost, accuracy


class Particle:  # 粒子
    # 初始化
    def __init__(self, para_max, para_min, dim, data, data_label, encoder_para, fault_num, encoder_iternum=20):

         # 粒子的位置
        self.fitnessValue_cost = None
        self.pos = [(np.random.rand() * (para_max[i] - para_min[i]))+ para_min[i]
                  for i in range(dim)]
        for i in range(len(self.pos)):
            if i != len(self.pos)-1:
                tmp_x = math.modf(self.pos[i])  # 分别取出整数部分和小数部分  输出tuple（1，2）2为整数部分，1为小数部分
                x_1 = int(tmp_x[1])  # 把整数部分转换为int
                if tmp_x[0] >= 0.5:
                    self.pos[i] = x_1 + 1
                else:
                    self.pos[i] = x_1

        self.bestPos = [0.0 for i in range(dim)]  # 粒子最好的位置
        cost, _ = fit_fun(self.pos, data, data_label, encoder_para, encoder_iternum, fault_num)  # 适应度函数值
        self.fitnessValue_cost = cost

    def set_pos(self, value):
        self.pos = value

    def get_pos(self):
        return self.pos

    def set_best_pos(self, value):
        self.bestPos = value

    def get_best_pos(self):
        return self.bestPos

    def set_fitness_value(self, value1):
        self.fitnessValue_cost = value1

    def get_cost_value(self):
        return self.fitnessValue_cost


class AOA:
    def __init__(self, size, aoa_iter_num, data, data_label, sda_para=False, cda_para=False, dda_para=False,
                 best_fitness_cost=float('Inf'),  dim=4, encoder_iternum=10, fault_num=10):

        self.dim = dim  # 粒子的维度
        self.size = size  # 粒子个数
        self.iter_num = aoa_iter_num  # aoa迭代次数
        self.encoder_iternum = encoder_iternum
        self.data = data
        self.data_label = data_label
        self.fault_num = fault_num
        self.u = 0.5
        self.alpha = 5
        self.MOA_min = 0.2
        self.MOA_max = 0.9
        self.best_fitness_cost = best_fitness_cost
        self.fitness_cost_list = []  # 每次迭代最优适应值
        self.position_list = []
        self.best_position = [0.0 for i in range(dim)]  # 种群最优位置


        # self.para_max = [447, 334, 250, 0.01]
        # self.para_min = [120, 75, 50, 0.002]
        self.para_max = []
        self.para_min = []
        input_unit = self.data.shape[1]
        out_unit = 3 * self.fault_num
        r = np.power(input_unit/out_unit, 1/3)
        min_unit1 = out_unit * np.power(r, 2)
        min_unit2 = out_unit * r
        self.para_min = [int(min_unit1), int(min_unit2), int(out_unit), 0.002]
        for i in range(self.dim-1):
            input_unit = np.sqrt(0.55 * np.power(input_unit, 2) + 3.31 * input_unit + 0.35) + 0.51
            self.para_max.append(int(input_unit))
        self.para_max.append(0.09)


        if sda_para == True:
            self.encoder_para = 'sda_para'
        elif cda_para == True:
            self.encoder_para = 'cda_para'
        elif dda_para == True:
            self.encoder_para = 'dda_para'


        # 初始化粒子
        self.Particle_list = []
        for i in range(self.size):
            p = Particle(self.para_max, self.para_min, self.dim, self.data,self.data_label,
                                       self.encoder_para, fault_num =self.fault_num ,encoder_iternum=self.encoder_iternum)
            self.position_list.append(p.pos)
            self.Particle_list.append(p)

        self.Particle_list.sort(key=lambda x:  x.fitnessValue_cost)
        self.best_position = copy.deepcopy(self.Particle_list[0].pos)
        self.best_fitness_cost = self.Particle_list[0].fitnessValue_cost


    def set_bestFitnessValue(self, value1):
        self.best_fitness_cost = value1

    def get_bestCostValue(self):
        return self.best_fitness_cost

    def set_bestPosition(self, value):
        self.best_position = value

    def get_bestPosition(self):
        return self.best_position


    # 更新位置
    def update_pos(self, part, C_iter):

        # 数学优化器概率（mop），此处MOP表示当前迭代时的函数值，α是一个敏感参数，定义了迭代过程中的开发精度，根据本文的实验，该精度固定为5。
        MOP = 1 - ((C_iter) ** (1 / self.alpha) / (self.iter_num) ** (1 / self.alpha))
        # 数学优化器加速（Math Optimizer Accelerated function）函数：MOA，
        # 此处MOA表示当前迭代的函数值，MOA_min，MOA_max表示加速函数的最小值和最大值， C_iter表示当前是第几次迭代
        MOA = self.MOA_min + C_iter * ((self.MOA_max - self.MOA_min) / self.iter_num)

        new_pos = [0] * self.dim
        best_pos = self.get_bestPosition()

        for i in range(self.dim):
            # （r1、r2和r3）生成[0,1]之间的随机值
            r1 = np.random.rand()
            r2 = np.random.rand()
            r3 = np.random.rand()

            UB = self.para_max[i]
            LB = self.para_min[i]

            '''
            AOA的探索算子在几个区域上随机探索搜索区域，并基于两种主要搜索策略（除法（Division）搜索策略和乘法搜索策略）寻找更好的解决方案，
            这两种策略在等式（3）中建模。
            对于r1>MOA（r1是一个随机数）的条件，该搜索阶段由数学优化器加速（MOA）函数进行调节。
            在该阶段（等式（3）中的第一条规则）中，第一个运算符（Division）的条件为r2<0。5，其他运算符（Multiplication）将被忽略，直到该操作员完成其当前任务。
            否则，第二个运算符（Multiplication）将参与执行当前任务，而不是Division（r2是一个随机数）。
            注意，元素考虑了一个随机比例系数，以产生更多的多样化过程并探索搜索空间的不同区域。
            '''
            # 位置更新方程
            if r1 > MOA:  # r1是一个随机数
                # Exploration phase 勘探阶段
                if r2 > 0.5:  # < ?
                    # Division， 应用除法数学运算符
                    new_pos[i] = best_pos[i] / (MOP + math.e) * ((UB - LB) * self.u + LB)
                else:
                    # Multiplication
                    new_pos[i] = best_pos[i] * MOP * ((UB - LB) * self.u + LB)

            else:
                # Exploitation phase
                if r3 > 0.5:  # < ?
                    # Subtraction
                    new_pos[i] = best_pos[i] - MOP * ((UB - LB) * self.u + LB)
                else:
                    # Addiction
                    new_pos[i] = best_pos[i] + MOP * ((UB - LB) * self.u + LB)

            if new_pos[i] < LB:
                new_pos[i] = LB  # 下限
            elif new_pos[i] > UB:
                new_pos[i] = UB  # 限制解的范围，上限

            # 单元数应该为int型，学习率可以是小数
            if 0 <= i < self.dim-1:
                if isinstance(new_pos[i], float):
                    p = math.modf(new_pos[i])  # 分别取出整数部分和小数部分  输出tuple（1，2）2为整数部分，1为小数部分
                    p_1 = int(p[1])  # 把整数部分转换为int
                    if p[0] >= 0.5:
                        new_pos[i] = p_1 + 1
                    else:
                        new_pos[i] = p_1

            # 限制范围
            if new_pos[i] < LB:
                new_pos[i] = LB
            elif new_pos[i] > UB:
                new_pos[i] = UB

        part.set_pos(new_pos)
        self.position_list.append(new_pos)
        cost, _ = fit_fun(part.get_pos(), self.data, self.data_label, self.encoder_para, train_step=self.encoder_iternum)
        if cost < part.get_cost_value():
            part.set_fitness_value(cost)
            part.set_best_pos(part.get_pos())  # 局部最优值

        if cost < self.get_bestCostValue():  # 全局最优值
            self.set_bestFitnessValue(cost)
            self.set_bestPosition(part.get_pos())


    def update(self):

        for i in range(self.iter_num):
            # 优化
            for part in self.Particle_list:
                self.update_pos(part, i)  # 更新位置

            costvalue = self.get_bestCostValue()
            self.fitness_cost_list.append(costvalue)  # 每次迭代完把当前的最优适应度存到列表
            if costvalue <= 1.0:
                break

        fina_pos = self.get_bestPosition()
        dir1 = './DHA_Bagging/AOA_out_para/'
        if not os.path.exists(dir1):
            os.mkdir(dir1)
        f = open('./DHA_Bagging/AOA_out_para/' + self.encoder_para + '.txt', 'w')
        print(fina_pos, file=f)
        f.close()

        print("PSO最优位置:" + str(self.get_bestPosition()))
        print("PSO cost最优解:" + str(self.fitness_cost_list))
        self.plot(self.fitness_cost_list, self.iter_num)
        self.plot_pos(self.position_list)

        return self.get_bestPosition()

    def plot(self, results, num):
        '''画图
        '''
        X = []
        Y = []
        for i in range(num):
            X.append(i + 1)
            Y.append(results[i])
        plt.plot(X, Y)
        plt.xlabel('Number of iteration', size=10)
        plt.ylabel('Value of cost', size=10)
        plt.title('encoder parameter optimization')
        plt.show()


    def plot_pos(self, pos_list):
        pos_list = np.array(pos_list)
        pca = PCA(n_components=2)
        pca_pos = pca.fit_transform(pos_list)

        plt.figure()
        for i in range(len(pos_list)):
            plt.scatter(pca_pos[i, 0], pca_pos[i, 1], c=i//self.size, marker='o')

        plt.legend()
        plt.show()







if __name__ == '__main__':
    data = shujuchuli.cut_samples(0)
    train_x, train_y, test_x, test_y = shujuchuli.make_datasets(data)
    train_x = train_x[:100,:]
    train_y = train_y[:100, :]
    test_x = test_x[:50,:]
    test_y = test_y[:50,:]

    train_y = np.argmax(train_y, 1).astype(np.int64)
    species = set(train_y)  # 数据可以分为几类  # unhashable type: 'numpy.ndarray'
    train_y = train_y.reshape(-1, 1)

    test_y = np.argmax(test_y, 1).astype(np.int64).reshape(-1, 1)

    size = 3
    iter_num = 2

    pso = AOA(size, iter_num, train_x, train_y, sda_para=True, encoder_iternum=30)
    best_pos = pso.update()
    print("PSO最优位置:" + str(best_pos))





