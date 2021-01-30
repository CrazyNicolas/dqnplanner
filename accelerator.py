import os
import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image
import time
import copy
from scipy.optimize import curve_fit

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pickle
import torchvision.transforms as T
# 请求问题类 包含问题的特征 及相应计算函数

loss_record = []
interval = 50.0
look_up = []
hit = []
running_duration = []
count = []
luck_rate = []


class POI:
    def __init__(self, longitude, latitude, trip_time, type, score, open, close, price):
        self.type = type
        self.longitude = longitude
        self.latitude = latitude
        self.trip_time = trip_time
        self.score = score
        self.open = open
        self.close = close
        self.price = price


poi = [POI(0, 0, 0, 0, "0", 0, 0, 0), ]
dist_line = 1
# cost_up = 0.2
# cost_lp = 0
# punish = 5
type_list = [13, 3, 2, 2, 1, 1, 1, 2, 3, 1, 6, 1, 9, 4, 1, 1, 7, 1, 2, 3, 3, 4, 6, 1, 2, 1, 36, 2, 3, 2, 4, 1, 2, 3, 1, 1, 4, 2, 1, 1, 34, 1, 1, 1, 1, 2, 5, 3, 1, 1, 3]


def read_poi():
    f = open("NGAin.txt", "r")
    n = int(f.readline())
    for i in range(n):
        s = f.readline()
        attr = s.split()
        [longitude, latitude, trip_time, Type, score, Open, close, price] = attr
        poi.append(POI(float(longitude), float(latitude), float(trip_time), Type, float(score), float(Open), float(close), float(price)))


def distance(u, v):
    if u == 0 or v == 0 or u == v:
        return 0
    xu = poi[u].longitude
    yu = 90 - poi[u].latitude
    xv = poi[v].longitude
    yv = 90 - poi[v].latitude
    return dist_line * 6371.004 * math.acos(math.sin(yu) * math.sin(yv) * math.cos(xu - xv) + math.cos(yu) * math.cos(yv)) * math.acos(-1) / 180.0


'''
:parameter
score_weight,
distance_weight,
variety_weight,
ub_budget,
lb_budget,
num_traveler,
avg_age,
max_age,
min_age,
duration,
season,
start_lon
start_lat
'''


class Request:
    def __init__(self, *args):
        self.features = [*args]

    def get_state(self, r):  # 计算requet之间的差值
        res = []
        for i in range(len(self.features)):
            res.append(self.features[i] - r.features[i])
        return torch.tensor([res], device=device, dtype=torch.float)

    def get_diff(self, r):  # 计算requet之间的差值
        res = []
        for i in range(len(self.features)):

            res.append(self.features[i] - r.features[i])

        return res


# 随机请求生成器
def req_generator():
    def generate():  # 1。 预算下届不少于预算上届的六成    2. 人数 1-6  3. 单位人头每天预算上届 700 + 200*标准正太分布随机数 4.min_age  < avg_age < max_age  5.广州位于东经112度bai57分至114度3分，北纬22度26分至23度56分
        all = 10
        score_weight = random.uniform(0, all)*all
        # all -= score_weight
        distance_weight = random.uniform(0, all)*all
        # all -= distance_weight
        variety_weight = random.uniform(0, all)*all
        sum = score_weight + distance_weight + variety_weight
        score_weight = score_weight/sum*10
        distance_weight = distance_weight/sum*10
        variety_weight = variety_weight/sum*10
        num_traveler = float(int(random.random()*10) % 6 + 1)  # 1-6
        duration = 3.0  # float(int(random.random()*10) % 7 + 1)  # 1-7
        ub_budget = float(int(np.random.normal(0, 0.3, 1)[0]*200 + 700))
        lb_budget = float(int(ub_budget*(0.6 + 0.4*random.random())))
        max_age = float(int(random.random()*100) % 80 + 20)  # 20-100
        min_age = max_age
        avg_age = max_age
        for i in range(int(num_traveler) - 1):
            age = int(random.random()*100) % 100 + 1  # 1-100
            avg_age += age
            max_age = max(max_age, age)
            min_age = min(min_age, age)
        avg_age = float(int(avg_age/num_traveler))
        season = int(random.random()*10) % 4 + 1  # 1-4
        start_lon = 112.95 + (114.05 - 112.95)*random.random()
        start_lat = 22.43 + (23.93-22.43)*random.random()
        return [distance_weight, score_weight, variety_weight, ub_budget, lb_budget, num_traveler, avg_age, max_age, min_age, duration, season, start_lon, start_lat]

    # 生成过程
    while True:
        f = generate()
        res = Request()
        for x in f:
            res.features.append(x)
        yield res


# t := <s,a,s',r>
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


# REPLAY MEMORY的逻辑后期可以再进行优化
class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DataBase:
    def __init__(self, max_size):
        self.table = {}
        self.max_size = max_size
        self.size = 0
        self.center = [0 for i in range(13)]

    def replace(self):
        # 随机删一个
        return self.table.popitem()

    def push(self, request, solution):
        for i in range(len(self.center)):
            self.center[i] *= self.size
        if self.size == self.max_size:
            popped = self.replace()
            self.size -= 1
            for i in range(len(self.center)):
                self.center[i] -= popped[0].features[i]

        self.table[request] = solution
        self.size += 1

        for i in range(len(self.center)):
            self.center[i] += request.features[i]
            self.center[i] = self.center[i]/self.size
        print("center:", self.center)

    def find(self, request):
        min = 99999
        result = None
        requesti = copy.deepcopy(request)
        requesti.features[0] /= 10
        requesti.features[1] /= 10
        requesti.features[2] /= 10
        requesti.features[3] /= 900
        requesti.features[4] /= 900
        requesti.features[5] /= 6
        requesti.features[6] /= 100
        requesti.features[7] /= 100
        requesti.features[8] /= 100
        requesti.features[10] /= 4
        requesti.features[11] /= 114.05
        requesti.features[12] /= 23.93
        for k, v in self.table.items():
            ki = copy.deepcopy(k)
            ki.features[0] /= 10
            ki.features[1] /= 10
            ki.features[2] /= 10
            ki.features[3] /= 900
            ki.features[4] /= 900
            ki.features[5] /= 6
            ki.features[6] /= 100
            ki.features[7] /= 100
            ki.features[8] /= 100
            ki.features[10] /= 4
            ki.features[11] /= 114.05
            ki.features[12] /= 23.93
            dis = np.sqrt(np.sum(np.square(requesti.get_diff(ki))))
            if dis < min:
                min = dis
                result = k

        return result

    def get_center(self):
        res = Request()
        for x in self.center:
            res.features.append(x)
        return res


class QNET(nn.Module):
    def __init__(self, feature_size):
        super(QNET, self).__init__()
        self.ln1 = nn.Linear(feature_size, 64)
        self.ln2 = nn.Linear(64, 16)
        self.out = nn.Linear(16, 2)

    def forward(self, x):
        x = self.ln1(x)
        x = nn.functional.sigmoid(x)
        x = self.ln2(x)
        x = nn.functional.sigmoid(x)
        return self.out(x.view(x.size(0), -1))


class QNET_Conv1D(nn.Module):
    def __init__(self,feature_size):
        super(QNET_Conv1D, self).__init__()
        self.conv1 = nn.Conv1d(feature_size,6,kernel_size=5)
        self.maxp = nn.MaxPool1d(kernel_size=2)
        self.ln1 = nn.Linear(372,128)
        self.ln2 = nn.Linear(128,32)
        self.ln3 = nn.Linear(32, 2)
    def forward(self,x):
        x = self.conv1(x)
        x = nn.ReLU(x)
        x = self.maxp(x)
        x = x.view(1,-1)
        x = self.ln1(x)
        x = nn.functional.sigmoid(x)
        x = self.ln2(x)
        x = nn.functional.sigmoid(x)
        x = self.ln3(x)
        x = nn.functional.sigmoid(x)
        return self.out(x.view(x.size(0), -1))


# 定义一些强化学习过程中需要的常量　
BATCH_SIZE = 128
GAMMA = 0.999  # reward衰减率
EPS_START = 0.9  # epsilon-greedy
EPS_END = 0.05
EPS_DECAY = 200
STEPS_DONE = 0  # 用于统计到目前为止走了多少步 是一个全局变量
epoch = 2500
db = DataBase(1024)
threshold_k = 0.9
current_req = next(req_generator())
# 初始化一个Replay Memory
memory = ReplayMemory(512)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_net = QNET(13).to(device)
eval_net = QNET(13).to(device)
eval_net.load_state_dict(train_net.state_dict())
eval_net.eval()

hour_per_km = 0.05
trip_time = 8
type_sum = 197

# 定义优化器为 RMSprop 这是原论文中使用的优化器
optimizer = optim.RMSprop(train_net.parameters())



def select_action(state):
    global STEPS_DONE
    rv = random.random()
    threshold = EPS_END + (EPS_START - EPS_END) * math.exp((-1) * STEPS_DONE / EPS_DECAY)
    STEPS_DONE += 1
    if rv > threshold:
        with torch.no_grad():
            return train_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(2)]], device=device, dtype=torch.long)


def GA(request):
    path = "GA "
    for i in range(5):
        path += str(request.features[i]) + " "
    path += str(request.features[9])
    data = []
    while len(data) == 0:
        print(path)
        f = os.popen(path)
        data = f.readlines()
        f.close()
    print(data[0])
    return [int(x) for x in data[0].split()]


# def GA_score(request, path):
#     Transit = 0
#     Score = 0
#     Occupied = 0
#     Type = 0
#     Cost = 0
#     Type_count = [0 for i in range(0, 55)]
#     trip_day = request.features[9]
#     upper_cost = request.features[3]
#     lower_cost = request.features[4]
#     rating = request.features[0]
#     transit = request.features[1]
#     diversity = request.features[2]
#     for i in range(len(path)):
#         Transit += hour_per_km * distance(path[i], path[min(i, len(path) - 1)])
#         if path[i] != 0:
#             Score += poi[path[i]].score
#             Occupied += poi[path[i]].trip_time
#             Cost += poi[path[i]].price
#             t = poi[path[i]].type
#             for j in range(0, len(t)):
#                 if t[j] == '0':
#                     Type_count[j] += 1
#     for i in Type_count:
#         Type += (1 - type_list[i] / type_sum) * i - (type_list[i] if i == 0 else 0)
#     f1 = -1 / trip_day * Transit
#     f2 = 1 / len(path) * Score
#     f3 = 1 / (len(path)) * Type
#     f4 = cost_up * (Cost - upper_cost / trip_day) if Cost > upper_cost / trip_day else (
#         cost_lp * (lower_cost / trip_day - Cost) if Cost < lower_cost / trip_day else 0)
#     return transit * f1 + rating * f2 + diversity * f3 - punish * (trip_time - Occupied - Transit) - f4

def GA_score(request, path):
    Transit = 0
    Score = 0
    Occupied = 0
    Type = 0
    Cost = 0
    number = 0
    Type_count = [0 for i in range(0, 51)]
    trip_day = request.features[9]
    upper_cost = request.features[3]
    lower_cost = request.features[4]
    rating = request.features[1]
    transit = request.features[0]
    diversity = request.features[2]
    for i in range(len(path)):
        Transit += hour_per_km * distance(path[i], path[min(i + 1, len(path) - 1)])
        if path[i] != 0:
            Score += poi[path[i]].score
            Occupied += poi[path[i]].trip_time
            Cost += poi[path[i]].price
            number += 1
            t = poi[path[i]].type
            for j in range(0, len(t)):
                if t[j] == '1':
                    Type_count[j] = 1
    for i in range(len(Type_count)):
        # Type += (1 - type_list[i] / type_sum) * Type_count[i] - (type_list[i] if Type_count[i] == 0 else 0)
        Type += Type_count[i]
    f1 = -1 / trip_day * Transit
    f2 = 1 / number * Score
    # f3 = 0.05 / (number) * Type
    f3 = Type / trip_day
    f4 = (Cost - upper_cost) / number if Cost > upper_cost else 0
    return transit * f1 + rating * f2 + diversity * f3 - (trip_day * trip_time - Occupied - Transit) * (transit + (rating + diversity) / trip_time) / 10 / trip_day - f4

# 当我们在DB中找到一个可以用做solution的和last_req最近的req  last_req 和 req 都对solution进行评分 计算二者比值p
# 设定阈值为k 那么 action若为 GA， 则reward固定为阈值 其余时候 如果p <  k, reward=0 重新使用GA 若p>=k 则reward = 1 * p


plt.figure(1)


def plot_durations():
    plt.figure(1)
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Percentage')
    l, = plt.plot(count, look_up, 'r', label='look_up')
    h, = plt.plot(count, hit, 'b', label='hit')
    plt.legend(handles=[l, h])
    # Take 50 episode averages and plot them too

    plt.pause(0.01)  # pause a bit so that plots are updated


def func(x, a, b):
#  y = a * log(x) + b
  y = x/(a*x+b)
  return y


def learning():
    if len(memory) < BATCH_SIZE:
        return

    # 抽取一些样本出来
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    next_state_batch = torch.cat(batch.next_state)

    Qvalues = train_net(state_batch).gather(1, action_batch)
    next_state_values = eval_net(next_state_batch).max(1)[0].detach()
    # print("next state:", next_state_values)
    # print("reward:", reward_batch)
    Yvalues = (next_state_values * GAMMA) + reward_batch.squeeze(1)

    # 计算LOSS
    loss = F.smooth_l1_loss(Qvalues, Yvalues.unsqueeze(1))
    # print("Qvalue:", Qvalues)
    # print("Yvlues:", Yvalues)
    # print("Yvlues uns:", Yvalues.unsqueeze(1))
    if i_epoch % interval == 0:
        loss_record.append(loss.item())
    # 下面使用优化器来优化
    optimizer.zero_grad()
    loss.backward()
    for param in train_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


# 在每次训练结束的时候不光要将网络参数权重存好 也要将replay memory还有DB存好 这样下次运行才能接着进行 相当于一个断点
def save_replay_mem():
    with open('RM.pkl','wb') as f:
        pickle.dump(memory,f)
def load_replay_mem():
    with open('RM.pkl', 'rb') as f:
        return pickle.load( f)
def save_DB():
    with open('DB.pkl', 'wb') as f:
        pickle.dump(db, f)
def load_DB():
    with open('DB.pkl', 'rb') as f:
        return pickle.load(f)


tot_look = 0
tot_hit = 0
tot_luck = 0
tot_time = 0
punish_rate = 1
decrease_rate = 0.9

if __name__ == '__main__':
    read_poi()
    look = 0.0
    h = 0.0
    t = 0.0
    luck = 0.0

    for i_epoch in range(epoch):
        print("epoch:", i_epoch)
        start = time.time()
        state = current_req.get_state(db.get_center())
        reward = 0
        # select_action
        action = select_action(state)
        if action == 0 or db.size == 0:  # GA运行
            solution = GA(current_req)
            db.push(current_req, solution)
            # reward = 0.5 * threshold_k
            reward = 0.6
        else:  # 查表
            look += 1
            h += 1
            req = db.find(current_req)
            solution = db.table[req]
            r1 = GA_score(current_req, solution)
            r2 = GA_score(req, solution)
            ratio = r1/r2
            reward = ratio
            if ratio > 1:
                luck += 1
            if ratio < threshold_k:
                h -= 1
                reward = ratio / 2 * punish_rate
                solution = GA(current_req)
                db.push(current_req, solution)
        next_req = next(req_generator())
        next_state = next_req.get_state(db.get_center())
        print(state, action, next_state, reward)
        memory.push(state, action, next_state, torch.tensor([[reward]], device=device, dtype=torch.float))
        current_req = next_req
        if i_epoch > BATCH_SIZE:
            punish_rate = punish_rate * decrease_rate
        a = time.time()
        learning()
        end = time.time()
        print("learning:", end - a)
        t += (end - start)
        if i_epoch % 10 == 0:
            eval_net.load_state_dict(train_net.state_dict())
        if i_epoch > 1 and i_epoch % interval == 0:
            tot_hit += h
            tot_look += look
            tot_luck += luck
            tot_time += t
            print(tot_look, tot_hit, tot_luck, tot_time)
            look_up.append(look/interval)
            hit.append(h/look if look > 0 else 0)
            running_duration.append(t / interval)
            luck_rate.append(luck/h if h > 0 else 0)
            t = 0.0
            look = 0.0
            h = 0.0
            luck = 0.0
            count.append(i_epoch)
            plot_durations()

        # todo:要根据循环次数 来记录一些数据信息 打印一些相应信息
    tot_hit += h
    tot_look += look
    tot_luck += luck
    tot_time += t
    print(tot_look, tot_hit, tot_luck, tot_time)
    look_up.append(look / interval)
    hit.append(h / look if look > 0 else 0)
    running_duration.append(t / interval)
    luck_rate.append(luck/h if h > 0 else 0)
        # todo:性能估量方面 ： 统计memory里面 选择数据库的比例 以及  选择数据库的那些request的hit rate要越来越高
    torch.save(train_net, 'testmodel.pkl')
    save_replay_mem()
    save_DB()


f = open("data_record.txt", "w")
f.writelines("Look up: ")
f.writelines(str(look_up))
f.writelines("\nhit: ")
f.writelines(str(hit))
f.writelines("\nRunning duration: ")
f.writelines(str(running_duration))
f.writelines("\nLoss: ")
f.writelines(str(loss_record))
f.writelines("\nLuck rate: ")
f.writelines(str(luck_rate))
f.writelines("\n")
f.close()

plt.figure(1)
plt.savefig("hit_look.png")

plt.figure(2)
plt.title('Time')
plt.xlabel('Episode')
plt.ylabel('avg_time')
xrange = len(running_duration) + 1
x0 = np.arange(1, xrange)
x1 = np.arange(1, xrange, 0.1)
plt.plot(x0, running_duration)
result = curve_fit(func, x0, running_duration, method='trf')
a, b = result[0]
y1 = x1/(a*x1+b)
plt.plot(x1, y1, 'red')
plt.savefig("Time.png")

plt.figure(3)
plt.title('Loss')
plt.xlabel('Episode')
plt.ylabel('loss')
xrange = len(loss_record) + 1
x0 = np.arange(1, xrange)
x1 = np.arange(1, xrange, 0.1)
plt.plot(x0, loss_record)
result = curve_fit(func, x0, loss_record, method='trf')
a, b = result[0]
y1 = x1/(a*x1+b)
plt.plot(x1, y1, 'red')
plt.savefig("Loss.png")

plt.figure(4)
plt.title('Luck')
plt.xlabel('Episode')
plt.ylabel('luck_rate')
xrange = len(luck_rate) + 1
x0 = np.arange(1, xrange)
x1 = np.arange(1, xrange, 0.1)
plt.plot(x0, luck_rate)
result = curve_fit(func, x0, luck_rate, method='trf')
a, b = result[0]
y1 = x1/(a*x1+b)
plt.plot(x1, y1, 'red')
plt.savefig("Luck_rate.png")

plt.figure(5)
plt.title('look_fit')
plt.xlabel('Episode')
plt.ylabel('look_rate')
xrange = len(look_up) + 1
x0 = np.arange(1, xrange)
x1 = np.arange(1, xrange, 0.1)
plt.plot(x0, look_up, 'blue')
result = curve_fit(func, x0, look_up, method='trf')
a, b = result[0]
y1 = x1/(a*x1+b)
plt.plot(x1, y1, 'red')
plt.savefig("look_fit.png")

plt.figure(6)
plt.title('hit_fit')
plt.xlabel('Episode')
plt.ylabel('hit_rate')
xrange = len(hit) + 1
x0 = np.arange(1, xrange)
x1 = np.arange(1, xrange, 0.1)
plt.plot(x0, hit, 'blue')
result = curve_fit(func, x0, hit, method='trf')
a, b = result[0]
y1 = x1/(a*x1+b)
plt.plot(x1, y1, 'red')
plt.savefig("hit_fit.png")

plt.show()

