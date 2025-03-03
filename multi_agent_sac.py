import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import collections
import ctypes
import os

from torch.distributions import Normal


# 定义 C++ 函数原型和数据结构
class OrderPlate(ctypes.Structure):
    _fields_ = [("thickness", ctypes.c_int), ("width", ctypes.c_int),("length", ctypes.c_int),
                ("arrival time", ctypes.c_int), ("deadline", ctypes.c_int), ("batch_id", ctypes.c_int)]

class Slab(ctypes.Structure):
    _fields_ = [("thickness", ctypes.c_int), ("width", ctypes.c_int), ("height", ctypes.c_int)]

class RollingMethod(ctypes.Structure):
    _fields_ = [("C1", ctypes.c_double), ("C2", ctypes.c_double), ("C3", ctypes.c_double), ("C4", ctypes.c_double)]

lib = ctypes.CDLL(os.path.abspath(os.path.join(os.path.dirname(__file__), 'cmake-build-debug/lib/libcpp_lib.so')))

# 定义 C++ 函数原型
lib.generate_data.argtypes = [ctypes.c_char_p, ctypes.c_int]
lib.generate_data.restype = ctypes.c_int

lib.read_data.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.POINTER(OrderPlate)), ctypes.POINTER(ctypes.POINTER(Slab)), ctypes.POINTER(ctypes.POINTER(RollingMethod))]
lib.read_data.restype = ctypes.c_void_p


# 设置 C++ 函数原型
lib.solveLinearProgramming.argtypes = [
    ctypes.POINTER(ctypes.c_double),  # benefit_pool
    ctypes.POINTER(ctypes.POINTER(OrderPlate)),  # scheme_pool
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int
]
lib.solveLinearProgramming.restype = ctypes.POINTER(ctypes.c_double)  # 返回对偶变量的指针





# 定义ReplayBuffer类
class ReplayBuffer:
    def __init__(self, buffer_size=10000):
        self.buffer = collections.deque(maxlen=buffer_size)  # 使用deque来实现FIFO
        self.buffer_size = buffer_size

    def add(self, state, actions, reward, next_state, done, order_boards, slabs, rolling_methods):
        # 将当前样本存储到replay buffer中
        self.buffer.append((state, actions, reward, next_state, done, order_boards, slabs, rolling_methods))

    def sample(self, batch_size):
        # 从replay buffer中随机抽取一批样本
        batch = random.sample(self.buffer, batch_size)
        state, actions, reward, next_state, done, order_boards, slabs, rolling_methods = zip(*batch)

        # 将样本打包成适合模型训练的格式
        return np.array(state), np.array(actions), np.array(reward), np.array(next_state), np.array(done), order_boards, slabs, rolling_methods

    def size(self):
        return len(self.buffer)


# Value Net
class ValueNet(nn.Module):
    def __init__(self, state_size, hidden_size):
        super(ValueNet, self).__init__()
        self.linear1 = nn.Linear(state_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, 1)

    def forward(self, state):
        x1 = F.relu(self.linear1(state))
        x2 = F.relu(self.linear2(x1))
        x3 = self.linear3(x2)

        return x3


# Q网络定义（SAC中使用）
class QNetwork(nn.Module):
    def __init__(self, obs_size, act_size, hidden_size=256):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(obs_size + act_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)  # 输出全局Q值

    def forward(self, state, actions):
        x = torch.cat([state, actions], dim=-1)  # 将状态和所有智能体的联合动作拼接
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)  # 输出全局Q值


class DeltaNet(nn.Module):
    def __init__(self, input_dim):
        super(DeltaNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)  # 输出一个标量，即温度参数 alpha

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        log_alpha = self.fc3(x)
        alpha = torch.sigmoid(log_alpha)  # 使用 Sigmoid 函数将输出限制在 [0, 1] 范围内
        return alpha


# 策略网络定义（每个智能体对应一个策略网络）
class PolicyNetwork(nn.Module):
    def __init__(self, obs_size, act_size,  device, hidden_size=256):
        super(PolicyNetwork, self).__init__()
        self.device = device
        self.fc1 = nn.Linear(obs_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, act_size)  # 输出动作
        self.mean_linear = nn.Linear(hidden_size, act_size)
        self.log_std_linear = nn.Linear(hidden_size, act_size)

    def forward(self, obs):
        x0 = F.relu(self.fc1(obs))
        x1 = F.relu(self.fc2(x0))
        x2 = self.fc3(x1)

        mean = self.mean_linear(x2)
        log_std = self.log_std_linear(x2)
        std = torch.exp(log_std)
        return mean,std  # 输出动作

    def action(self, observation):
        observation = torch.FloatTensor(observation).to(self.device)
        mean, log_std = self.forward(observation)
        std = log_std.exp()
        normal = Normal(mean, std)
        z = normal.sample()
        action = torch.tanh(z).detach().numpy()

        return action

    def evaluate(self, obs,epsilon=1e-6):
        mean, std = self.forward(obs)
        normal = Normal(mean, std)
        noise = Normal(0,1)

        z=noise.sample()
        action = torch.tanh(mean + std * z.to(self.device))
        log_prob = normal.log_prob(mean + std * z.to(self.device)) - torch.log(1 - action.pow(2) + epsilon)
        log_prob = log_prob.sum(1,keepdim=True)

        return action, log_prob


# Pointer Network
class PointerNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, num_selected):
        super(PointerNetwork, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_selected = num_selected

        # 定义网络结构
        self.encoder = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.attn_layer = nn.Linear(hidden_size, 1)

    def forward(self, x):
        """
        输入 x 形状： (batch_size, sequence_length, input_size)
        """
        # 通过LSTM编码器
        lstm_out, _ = self.encoder(x)
        attn_weights = self.attn_layer(lstm_out)
        attn_weights = F.softmax(attn_weights, dim=1)  # 对输出进行softmax，获得注意力权重

        return attn_weights


    def action(self, x):
        attn_weights = self.forward(x)
        # 根据注意力权重选择最重要的m个订单板
        selected_indices = torch.topk(attn_weights, self.num_selected, dim=1).indices
        return selected_indices


    def evaluate(self, x):
        attn_weights = self.forward(x)
        probs, selected_indices = torch.topk(attn_weights, self.num_selected, dim=1).indices
        return probs, selected_indices


# SAC Agent类定义
class SACAgent:
    def __init__(self, obs_size, agent_index, act_size, input_size, hidden_size, num_selected, lr):
        self.policy = PolicyNetwork(obs_size, act_size)
        if agent_index == 0 or agent_index == 1:
            self.policy = PointerNetwork(input_size, hidden_size, num_selected)

        self.optimizer_policy = optim.Adam(self.policy.parameters(), lr=lr)




    def select_action(self, agent_index, joint_observation):
        # 策略网络根据观察选择动作
        action = self.policy.action(obs=joint_observation[agent_index])
        return action

    def evaluate_action(self,agent_index,joint_observation):
        action, log_prob = self.policy.evaluate(obs=joint_observation[agent_index])
        return action, log_prob

    def update(self, new_log_prob, q_value):

        return 0





# 环境类，管理多个智能体
class MultiAgentEnv:
    def __init__(self):
        self.num_selected = 30
        self.n_agents = 4
        self.lr = 0.00002
        self.max_num_slabs = 20
        self.max_num_rolling_methods = 15
        self.obs_size_list = [0, 3*self.num_selected, 3* self.max_num_slabs, 3*self.max_num_rolling_methods]
        self.act_size_list = [self.num_selected, 2, 1, 1]
        self.input_size = 8
        self.hidden_size = 256
        self.agents = [SACAgent(self.obs_size_list[i], self.act_size_list[i],self.input_size,self.hidden_size,self.num_selected,self.lr) for i in range(self.n_agents)]

        self.state_size = 1000
        self.observation_size = 2000
        self.hidden_size = 32
        self.action_size = 30
        self.gamma = 0.00001
        self.tau = 0.005
        self.beta = 0.0001
        self.T = 2000
        self.order_plates = None
        self.order_plates_pool = []
        self.slabs = None
        self.rolling_methods = None
        self.obj_weights = [0.9, 0.1]
        self.v = 0.4
        self.scheme_pool = []
        self.benefit_pool = []

        self.V = 1.0

        self.target_q_net = ValueNet(self.state_size,self.hidden_size)
        self.q_net = QNetwork(self.observation_size,self.action_size,self.hidden_size)
        self.delta_net = DeltaNet(2)

        # Initialize the optimizer
        self.target_q_net_optimizer = optim.Adam(self.target_q_net.parameters(), lr=self.lr)
        self.q_net_optimizer = optim.Adam(self.q_net.parameters(), lr=self.lr)
        self.delta_net_optimizer = optim.Adam(self.delta_net.parameters(), lr=self.lr)
        self.policy_optimizer = optim.Adam(list(self.agents[0].policy.parameters()) + list(self.agents[1].policy.parameters()) +
                               list(self.agents[2].policy.parameters()) + list(self.agents[3].policy.parameters()), lr=self.lr)

    def reset(self, instance_num):
        data_filename = "instance"+str(instance_num)
        self.generate_new_DSDI(data_filename)
        max_volume_slab = max(self.slabs, key=lambda slab: slab.thickness*slab.width*slab.length)
        self.V = self.v * max_volume_slab
        joint_observation = self.construct_observation(self.order_plates[0],self.slabs,self.rolling_methods)
        global_state = self.construct_state(joint_observation)
        self.order_plates_pool = self.order_plates[0]

        return joint_observation, global_state

    def step(self, joint_action, selected_order_plates, t):
        reward, produced_order_plate_indices = self.calculate_reward(joint_action, selected_order_plates, t)
        self.generate_possible_scheme(t)
        remain_order_plates = [ord_pla for ord_pla in self.order_plates_pool if ord_pla.index not in produced_order_plate_indices]
        self.order_plates_pool = remain_order_plates
        for ord_pla in self.order_plates[t]:
            self.order_plates_pool.append(ord_pla)
        next_joint_observation = self.construct_observation(self.order_plates_pool,self.slabs,self.rolling_methods)
        next_global_state = self.construct_state(next_joint_observation)
        done = False

        return next_global_state, next_joint_observation, reward, done

    def generate_possible_scheme(self, t):
        used_order_plates = self.order_plates_pool
        rand_list = [0 if random.randint(1,10) < 5 else 1 for i in range(len(used_order_plates))]
        for i in range(len(used_order_plates)):
            if rand_list[i] == 0:
                used_order_plates.remove(used_order_plates[i])
                rand_list.remove(rand_list[i])
                i = i -1

        joint_observation = self.construct_observation(used_order_plates,self.slabs,self.rolling_methods)
        joint_action = []
        selected_order_plates = []

        for i in range(4):
            if i == 1:
                for j in range(len(joint_action[0])):
                    selected_order_plates.append(self.order_plates[joint_action[0][j]])
                self.update_observation(selected_order_plates,joint_observation)
            action, lob_prob = self.agents[i].select_action(i,joint_observation)
            joint_action.append(action)

        selected_slab = self.slabs[joint_action[2]]
        selected_rolling_method = self.rolling_methods[joint_action[3]]
        reward, produced_order_plate_indices = self.calculate_reward(joint_action,selected_order_plates,t)
        self.scheme_pool[t][selected_slab.index][selected_rolling_method.index] = [self.order_plates[i] for i in produced_order_plate_indices]
        self.benefit_pool[t][selected_slab.index][selected_rolling_method.index] = reward



    def calculate_reward(self, joint_action, selected_order_plates, t):
        selected_slab = self.slabs[joint_action[2]]
        selected_rolling_method = self.rolling_methods[joint_action[3]]
        produced_order_plate_indices = lib.TwoCDP(selected_order_plates,selected_slab,selected_rolling_method)
        total_volume = sum(order_plate.thickness * order_plate.width * order_plate.length for order_plate in selected_order_plates if order_plate.index in produced_order_plate_indices)
        reward = (self.obj_weights[0] * total_volume +
                  self.obj_weights[1] * (selected_slab.thickness * selected_slab.width * selected_slab.length - total_volume))
        self.scheme_pool[t][selected_slab.index][selected_rolling_method.index] = [self.order_plates[i] for i in produced_order_plate_indices]
        self.benefit_pool[t][selected_slab.index][selected_rolling_method.index] = reward
        return reward, produced_order_plate_indices


    def call_dual_factor(self):
        T1, J1, G1, Q1 = self.benefit_pool.shape
        result_ptr = lib.solveLinearProgramming(self.benefit_pool, self.scheme_pool, T1, J1, G1, Q1)

        dual_values = np.ctypeslib.as_array(result_ptr, shape=(T1 * J1 * G1 * Q1,))

        return dual_values

    def calculate_policy_weight(self,dual_values, num_order_plates, num_slabs, num_rolling_methods, T1):
        omega = [0, 0, 0, 0]

        temp1 = [num_order_plates-1, num_order_plates, num_order_plates+T1*num_slabs, len(dual_values)]

        for i in range(temp1[0]):
            omega[0] += dual_values[i]
        omega[0] = omega[0]*self.num_selected/num_order_plates

        for i in range(temp1[0]):
            omega[1] += dual_values[i]
        omega[1] = omega[1]* 2/ num_order_plates

        for tau in range(T1):
            for j in range(num_slabs):
                omega[2] += dual_values[temp1[1]+1+tau*j]
        omega[2] = omega[2]/ (num_slabs * (T1 +1 ))

        for tau in range(T1):
            for g in range(num_rolling_methods):
                omega[3] += dual_values[temp1[2] + 1 + tau * g]
        omega[3] = omega[3]/ (num_rolling_methods*(T1+1))

        policy_weight = [0, 0, 0, 0]
        for i in range(4):
            policy_weight[i] = omega[i]/sum(omega)

        return  policy_weight


    def generate_new_DSDI(self, data_filename):
        n_batch = random.randint(170,190)
        lib.generate_data(data_filename.encode('utf-8'),n_batch)
        lib.read_data(data_filename.encode('utf-8'), self.order_plates, self.slabs, self.rolling_methods)

    def construct_state(self, joint_observation):
        global_state = []
        for i in range(len(joint_observation)):
            if i == 1:
                continue
            for j in range(len(joint_observation[i])):
                global_state.append(joint_observation[i][j])

        return global_state

    def construct_observation(self, current_order_plates, slabs, rolling_methods):
        joint_observation = []
        observation1 = []
        for i in range(len(current_order_plates)):
            observation1.append(current_order_plates[i].thickness)
            observation1.append(current_order_plates[i].width)
            observation1.append(current_order_plates[i].length)

        observation2 = []

        observation3 = []
        for j in range(len(slabs)):
            observation3.append(slabs[j].thickness)
            observation3.append(slabs[j].width)
            observation3.append(slabs[j].length)

        observation4 = []
        for g in range(len(rolling_methods)):
            observation4.append(rolling_methods[g].C1)
            observation4.append(rolling_methods[g].C2)
            observation4.append(rolling_methods[g].C3)
            observation4.append(rolling_methods[g].C4)

        joint_observation.append(observation1)
        joint_observation.append(observation2)
        joint_observation.append(observation3)
        joint_observation.append(observation4)

        return joint_observation

    def update_joint_observation(self, selected_order_plates, joint_observation):
        for i in range(len(selected_order_plates)):
            joint_observation[1].append(selected_order_plates[i].thickness)
            joint_observation[1].append(selected_order_plates[i].width)
            joint_observation[1].append(selected_order_plates[i].length)

        return joint_observation

    def update_order_plates(self, current_order_plates, produced_order_plates):
        produced_indexes = {order_plate.index for order_plate in produced_order_plates}
        current_order_plates = [order_plate for order_plate in current_order_plates if order_plate.index not in produced_indexes]

        return current_order_plates

    def collect_order_plates(self, current_order_plates, new_order_plates):
        for i in range(len(new_order_plates)):
            current_order_plates.append(new_order_plates[i])

        return current_order_plates

    def update(self, replay_buffer, batch_size, t):
        global_state_batch, joint_observation_batch, joint_action_batch, reward_batch, next_state_batch, done_batch = replay_buffer.sample(batch_size)

        new_joint_action = []
        selected_order_plates=[]
        new_log_prob = []

        for i in range(4):
            if i == 1:
                for j in range(len(new_joint_action[0])):
                    selected_order_plates.append(self.order_plates[new_joint_action[0][j]])
                self.update_observation(selected_order_plates,joint_observation_batch)
            action, lob_prob = self.agents[i].select_action(i,joint_observation_batch)
            new_joint_action.append(action)
            new_log_prob.append(lob_prob)


        value = self.target_q_net(global_state_batch)
        new_q_value = self.q_net(global_state_batch,new_joint_action)
        next_value = new_q_value-sum(new_log_prob)
        value_loss = F.mse_loss(value, next_value)


        # 计算联合Q值
        q_value = self.q_net(global_state_batch, joint_action_batch)

        with torch.no_grad():
            target_value = self.target_q_net(next_state_batch)
            target_q_value = reward_batch + self.gamma * target_value * done_batch

        # 计算Q值损失
        loss_q = F.mse_loss(q_value, target_q_value)

        self.optimizer_q1.zero_grad()
        loss_q.backward()
        self.optimizer_q1.step()

        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        delta = self.delta_net(torch.tensor([q_value,self.V]))
        # 联合策略网络的损失函数
        policy_loss = torch.mean(sum(new_log_prob) - (1 / self.beta) * ((1+delta) * q_value - delta * (self.T-t+1)*self.V))

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # for agent in self.agents:
        #     agent.update(new_log_prob, q_value)

        delta_loss = self.delta * (q_value - self.delta * (self.T-t+1)*self.V).detach().mean()
        self.delta_optimizer.zero_grad()  # 清除上一步的梯度
        delta_loss.backward()  # 计算新的梯度
        self.delta_optimizer.step()  # 更新神经网络的参数


# 训练多智能体
def train_multi_agent():
    # 初始化ReplayBuffer
    replay_buffer = ReplayBuffer(buffer_size=10000)
    env = MultiAgentEnv()
    num_episodes = 200
    batch_size = 512

    for episode in range(num_episodes):
        joint_observation, global_state = env.reset(episode)
        done = False
        total_reward = 0
        t = 0

        while not done:
            joint_action = []
            selected_order_plates = []
            env.scheme_pool.append([[[] for g in len(env.rolling_methods)] for j in len(env.slabs)])
            env.benefit_pool.append([[[] for g in len(env.rolling_methods)] for j in len(env.slabs)])
            # 获取每个智能体的行动和选择的订单板索引
            for i in env.n_agents:
                if i == 1:
                    for j in range(len(joint_action[0])):
                        selected_order_plates.append(env.order_plates[joint_action[0][j]])
                    env.update_observation(selected_order_plates,joint_observation)
                action = env.agents[i].select_action(i,joint_observation)
                joint_action.append(action)


            # 将当前状态、动作、奖励、下一个状态等存入replay buffer
            next_global_state, next_joint_observation, reward, done, _ = env.step(joint_action, selected_order_plates, t)
            replay_buffer.add(global_state, joint_observation, joint_action, reward, next_global_state, done)

            # 每隔一定步骤，进行策略更新
            if replay_buffer.size() >= batch_size:
                env.update(replay_buffer,batch_size,t)


            total_reward += reward
            global_state = next_global_state
            joint_observation = next_joint_observation
            t = t + 1
        print(f"Episode {episode} | Total Reward: {total_reward}")
