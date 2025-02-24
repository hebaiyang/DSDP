import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import collections
import ctypes
import os

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


# 定义状态、观察和策略
class MultiAgentState:
    def __init__(self, order_boards, slabs, rolling_methods):
        self.order_boards = order_boards  # 所有订单板
        self.slabs = slabs  # 所有板坯
        self.rolling_methods = rolling_methods  # 所有轧制方法

    def get_state(self, remaining_order_boards, current_order_boards):
        # 状态是剩余订单板和当前订单板的并集，包含厚度、宽度、长度
        return np.concatenate([remaining_order_boards, current_order_boards], axis=0)

    def get_observation(self, agent_type, I_t, slab_data=None, rolling_method_data=None):
        if agent_type == "slab":
            # 选择板坯的智能体的观察：所有板坯的特征
            return np.array([[slab.thickness, slab.width, slab.length] for slab in slab_data])
        elif agent_type == "rolling_method":
            # 选择轧制方法的智能体的观察：所有轧制方法的系数
            return np.array([method.coefficient for method in rolling_method_data])
        elif agent_type == "order_board_selection":
            # 选择m个订单板的智能体的观察：当前I_t中的订单板特征
            return np.array([[order.thickness, order.width, order.length] for order in I_t])
        elif agent_type == "order_board_k1_k2":
            # 选择k1, k2订单板的智能体的观察：所有订单板的特征
            return np.array([[order.thickness, order.width, order.length] for order in I_t])


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


# 策略网络定义（每个智能体对应一个策略网络）
class PolicyNetwork(nn.Module):
    def __init__(self, obs_size, act_size, hidden_size=256):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(obs_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, act_size)  # 输出动作

    def forward(self, obs):
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        return self.fc3(x)  # 输出动作


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

        # 根据注意力权重选择最重要的m个订单板
        selected_indices = torch.topk(attn_weights, self.num_selected, dim=1).indices
        return selected_indices


# SAC Agent类定义
class SACAgent:
    def __init__(self, obs_size, act_size, input_size, hidden_size, num_selected, lr=1e-3, gamma=0.99, tau=0.005, beta=0.01):
        self.policy = PolicyNetwork(obs_size, act_size)
        self.q1 = QNetwork(obs_size, act_size)
        self.target_q1 = QNetwork(obs_size, act_size)

        # 添加指针网络以选择订单板
        self.pointer_network = PointerNetwork(input_size, hidden_size, num_selected)

        self.optimizer_policy = optim.Adam(self.policy.parameters(), lr=lr)
        self.optimizer_q1 = optim.Adam(self.q1.parameters(), lr=lr)

        # 复制q1到target_q1
        self.target_q1.load_state_dict(self.q1.state_dict())

        self.gamma = gamma
        self.tau = tau
        self.beta = beta

    def select_action(self, angent_index, joint_observation):
        """
        # 根据订单板选择的索引，选择具体的订单
        # """
        # if agent_type == "slab":
        #     observation = torch.Tensor([slab.thickness, slab.width, slab.length] for slab in slabs)
        # elif agent_type == "rolling_method":
        #     observation = torch.Tensor([method.coefficient for method in rolling_methods])
        # elif agent_type == "order_board_selection":
        #     observation = torch.Tensor([[order.thickness, order.width, order.length] for order in order_boards])
        # else:
        #     # 选择订单板k1, k2
        #     observation = torch.Tensor([[order.thickness, order.width, order.length] for order in order_boards])

        # 策略网络根据观察选择动作
        if angent_index == 1:
            action = self.pointer_network(joint_observation[angent_index])
        else:
            action = self.policy(joint_observation[angent_index])
        return action

    def update(self, replay_buffer, batch_size):
        obs, action, reward, next_obs, done, order_boards, slabs, rolling_methods = replay_buffer.sample(batch_size)

        # 计算联合Q值
        joint_actions = torch.cat(action, dim=-1)  # 联合动作
        q_value = self.q1(obs, joint_actions)

        with torch.no_grad():
            next_action = self.policy(next_obs)
            target_q_value = self.target_q1(next_obs, next_action)
            target_q_value = reward + self.gamma * target_q_value * (1 - done)

        # 计算Q值损失
        loss_q = F.mse_loss(q_value, target_q_value)

        self.optimizer_q1.zero_grad()
        loss_q.backward()
        self.optimizer_q1.step()

        # 联合策略网络的损失函数
        joint_action = torch.cat(action, dim=-1)  # 联合动作
        policy_loss = 0
        for i in range(4):
            # 每个智能体的动作在联合动作中的损失
            pi = self.policy(obs)  # 策略网络输出
            policy_loss += torch.mean(torch.log(pi) - (1 / self.beta) * q_value)

        self.optimizer_policy.zero_grad()
        policy_loss.backward()
        self.optimizer_policy.step()

        # 执行软更新
        self.soft_update(self.target_q1, self.q1)

    def soft_update(self, target, source):
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(self.tau * source_param.data + (1.0 - self.tau) * target_param.data)


# 环境类，管理多个智能体
class MultiAgentEnv:
    def __init__(self, agents, n_agents, env_params):
        self.agents = agents
        self.env_params = env_params
        self.order_plates = None
        self.slabs = None
        self.rolling_methods = None
        self.obj_weights = [0.9, 0.1]

    def reset(self, instance_num):
        data_filename = "instance"+str(instance_num)
        self.generate_new_DSDI(data_filename)
        joint_observation = self.construct_observation(self.order_plates[0],self.slabs,self.rolling_methods)
        global_state = self.construct_state(joint_observation)

        return joint_observation, global_state

    def step(self, joint_action, selected_order_plates, step_index):
        next_joint_observation = self.construct_observation(selected_order_plates[step_index],self.slabs,self.rolling_methods)
        next_global_state = self.construct_state(next_joint_observation)
        reward = self.calculate_reward(joint_action, selected_order_plates)
        done = False
        return next_global_state, next_joint_observation, reward, done

    def calculate_reward(self, joint_action, selected_order_plates):
        selected_slab = self.slabs[joint_action[2]]
        selected_rolling_method = self.rolling_methods[joint_action[3]]
        produced_order_plate_indices = lib.TwoCDP(selected_order_plates,selected_slab,selected_rolling_method)
        total_volume = sum(order_plate.thickness * order_plate.width * order_plate.length for order_plate in selected_order_plates if order_plate.index in produced_order_plate_indices)
        reward = (self.obj_weights[0] * total_volume +
                  self.obj_weights[1] * (selected_slab.thickness * selected_slab.width * selected_slab.length - total_volume))
        # 计算联合奖励：基于所有智能体的动作来计算奖励
        return reward, produced_order_plate_indices


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


# 训练多智能体
def train_multi_agent(agents, env, num_episodes, batch_size, hyperparams):
    # 初始化ReplayBuffer
    replay_buffer = ReplayBuffer(buffer_size=10000)

    for episode in range(num_episodes):
        joint_observation, global_state = env.reset(episode)
        done = False
        total_reward = 0

        lr = hyperparams['lr']
        agamma = hyperparams['gamma']
        tau = hyperparams['tau']
        beta = hyperparams['beta']

        while not done:
            joint_action = []
            selected_order_plates = []
            # 获取每个智能体的行动和选择的订单板索引
            for i in env.n_agents:
                if i == 1:
                    for j in range(len(joint_action[0])):
                        selected_order_plates.append(env.order_plates[joint_action[0][j]])
                    env.update_observation(selected_order_plates,joint_observation)
                joint_action.append(agents[i].selection_action(i,joint_observation))


            # 将当前状态、动作、奖励、下一个状态等存入replay buffer
            next_global_state, reward, done, _ = env.step(joint_action, selected_order_plates, episode)
            replay_buffer.add(global_state, joint_observation, joint_action, reward, next_global_state, done)

            # 每隔一定步骤，进行策略更新
            if replay_buffer.size() >= batch_size:
                global_state_batch, joint_observation_batch, joint_action_batch, reward_batch, next_state_batch, done_batch = replay_buffer.sample(batch_size)

                for agent in agents:
                    agent.update(replay_buffer, batch_size)

            total_reward += reward
            global_state = next_global_state

        print(f"Episode {episode} | Total Reward: {total_reward}")
