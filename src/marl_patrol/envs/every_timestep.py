"""Every-timestep patrolling environment.

Every agent selects an action at every simulation step via :meth:`step`, which
returns the next observation together with a reward (idleness-based on a node,
a small ``-0.1`` penalty while in transit on an edge).
"""

import imageio
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from sklearn.neighbors import NearestNeighbors

from marl_patrol.envs._compat import gym
from marl_patrol.envs.render_utils import grab_rgb_frame

# Parameters
ALPHA = 1.1  # distance threshold (in step lengths) for "close enough" to a node

np.set_printoptions(precision=4, suppress=True)


def generate_points_with_min_distance(n, size, min_dist):
    # compute grid size based on number of points
    width_ratio = size[1] / size[0]
    num_y = np.int32(np.sqrt(n / width_ratio)) + 1
    num_x = np.int32(n / num_y) + 1

    # create regularly spaced neurons
    x = np.linspace(0.0, size[1], num_x, dtype=np.float32)
    y = np.linspace(0.0, size[0], num_y, dtype=np.float32)

    coords = np.stack(np.meshgrid(x, y), -1).reshape(-1, 2)

    # compute spacing
    init_dist = np.max((x[1] - x[0], y[1] - y[0]))
    max_movement = (init_dist - min_dist) / 2

    # perturb points
    low = np.ones(len(coords)) * (-np.pi)
    high = np.ones(len(coords)) * (np.pi)
    rad = np.random.rand(len(coords)) * max_movement

    # left
    left = np.logical_and(
        coords[:, 0] < max_movement,
        np.logical_and(coords[:, 1] >= max_movement, coords[:, 1] <= size[0] - max_movement),
    )
    low[left], high[left] = -np.pi / 2, np.pi / 2

    # top left
    topleft = np.logical_and(coords[:, 0] < max_movement, coords[:, 1] > size[0] - max_movement)
    low[topleft], high[topleft] = -np.pi / 2, 0

    # top
    top = np.logical_and(
        np.logical_and(coords[:, 0] >= max_movement, coords[:, 0] <= size[1] - max_movement),
        coords[:, 1] > size[0] - max_movement,
    )
    low[top], high[top] = -np.pi, 0

    # top right
    topright = np.logical_and(
        coords[:, 0] > size[1] - max_movement, coords[:, 1] > size[0] - max_movement
    )
    low[topright], high[topright] = -np.pi, -np.pi / 2
    # right
    right = np.logical_and(
        coords[:, 0] > size[1] - max_movement,
        np.logical_and(coords[:, 1] >= max_movement, coords[:, 1] <= size[0] - max_movement),
    )
    low[right], high[right] = -3 * np.pi / 2, -np.pi / 2
    # bottom right
    bottomright = np.logical_and(coords[:, 0] > size[1] - max_movement, coords[:, 1] < max_movement)
    low[bottomright], high[bottomright] = np.pi / 2, np.pi
    # bottom
    bottom = np.logical_and(
        np.logical_and(coords[:, 0] >= max_movement, coords[:, 0] <= size[1] - max_movement),
        coords[:, 1] < max_movement,
    )
    low[bottom], high[bottom] = 0, np.pi
    # bottom left
    bottomleft = np.logical_and(coords[:, 0] < max_movement, coords[:, 1] < max_movement)
    low[bottomleft], high[bottomleft] = 0, np.pi / 2

    heading = np.random.uniform(low=low, high=high)

    noises = np.zeros_like(coords)
    noises[:, 0] = np.cos(heading) * rad
    noises[:, 1] = np.sin(heading) * rad

    coords += noises

    randomIndex = np.random.choice(np.arange(len(coords)), size=n, replace=False)
    coords = coords[randomIndex]

    return coords


def findNodeIndexByPos(envMap, pos):
    """
    Find whether the coordinate 'pos' is on one of the graph nodes
    If match, return the index of this node
    """

    assert isinstance(envMap, nx.Graph)
    for n, v in envMap.nodes().items():
        if (v["pos"] == pos).all():  # if the coord matches certain node
            return n  # return the index of this point
    return None


def checkOnInterval(P, P1, P2):
    """
    Check if P, P1, P2 are on the same line
    """

    v1 = P1 - P
    v2 = P2 - P
    dot = np.dot(v1, v2)
    if dot == 0:
        return False
    else:
        dot /= np.linalg.norm(v1) * np.linalg.norm(v2)
        dot = np.clip(dot, -1.0, 1.0)
        theta = np.arccos(dot)
        if theta >= 0.99 * np.pi:
            return True
        else:
            return False


def findEdgeByPos(envMap, pos, next_target_ind, targetCoords):
    """
    Find whether current position coord is on an edge.
    Args:
        pos: current coord
        next_target_ind: index of next destination (index determined once the env is generated)
        targetCoords: All targets' coords (fixed once the env is generated)
    Return:
        [index of node behind me, index of node ahead of me]
    """
    next_target_pos = targetCoords[next_target_ind]  # 下一target坐标
    next_target_nbr_ind = list(envMap.neighbors(next_target_ind))  # 与下一个target相邻的节点
    for ind in next_target_nbr_ind:
        nbr_pos = targetCoords[ind]
        if checkOnInterval(
            pos, nbr_pos, next_target_pos
        ):  # 检查当前坐标，下一target和下一target邻居坐标三点是否共线。其中必有一个是该点所在边的另一个端点
            return [ind, next_target_ind]  # 身后的端点的index, 前方目标端点的index


class Agent:
    def __init__(self, ID, pos, envMap, targetCoords, idleness, step_length=0.05):
        super().__init__()
        self.ID = ID
        self.pos = pos
        self.localMap = envMap
        self.localTargetCoords = targetCoords
        self.localIdleness = idleness
        self.isOnNode = True
        self.prev_target_ind = None
        self.next_target_ind = None
        self.step_length = step_length
        self.cachedLocalIdleness = []
        self.cachedNeighborObs = []
        self._elapsed_steps = 0
        self.finished = False
        self._initAgent()

    @property
    def elapsed_steps(self):
        return self._elapsed_steps

    def _initAgent(self):
        self.prev_target_ind = findNodeIndexByPos(
            self.localMap, self.pos
        )  # 此时的pos是上一时刻的target,取得其index
        nbr_ind = list(
            self.localMap.neighbors(self.prev_target_ind)
        )  # 邻居的prev_target跟↑一样 findnode
        self.next_target_ind = np.random.choice(nbr_ind)  # 初始时刻下一个目标点从邻居里随便取

    def fuse(self):
        """
        把自己的localIdlenessMap[shape(10, )]和自己收集的邻居的(不包括自己)LocalIdlenessMap(即cachedLocalIdleness)[shape(10, 3)]拼一起(shape(10, 4))
        然后竖着逐项比较取最小, 返回更新后的localIdlenessMap[shape(10, )]
        """
        cachedLocalIdleness = [self.localIdleness] + self.cachedLocalIdleness
        cachedLocalIdleness = np.array(cachedLocalIdleness)
        self.localIdleness = np.min(cachedLocalIdleness, axis=0)
        return self.localIdleness

    def get_target_coords(self, relative=True):
        """
        Get targets coordinates.
        """

        if relative:  # 把绝对坐标转成相对坐标
            target_coords = self.localTargetCoords - self.pos
        # absolute observation
        else:
            target_coords = (
                self.localTargetCoords
            )  # array, size(10, 2) [[x1, y1], [x2, y2], ... , [xn, yn]]
        return target_coords

    def getNbrObs(self, relative=True):
        """
        Get agent states (num_agent, 4)
        Append myself to the neighbor states
        Convert [(xx, xx), (xx, xx)] to numpy
        Return all agent states (num_agent, 4)
        """
        cachedNeighborObs = [
            (self.pos, self.localTargetCoords[self.next_target_ind])
        ] + self.cachedNeighborObs  # 把自己的状态加进去 列表相加效果类似append但是生成了一个新的对象
        agent_states = np.zeros((len(cachedNeighborObs), 4))  # 初始化neighbor的obs为全0
        # relative observation w.r.t. myself
        if relative:  # 把绝对坐标转成相对坐标
            for i, nbr in enumerate(cachedNeighborObs):
                pos, next = nbr
                agent_states[i, :2] = pos - self.pos
                agent_states[i, 2:4] = next - self.pos
        # absolute observation
        else:
            for i, nbr in enumerate(cachedNeighborObs):
                pos, next = nbr
                agent_states[i, :2] = pos
                agent_states[i, 2:4] = next
        return agent_states

    def move(self):
        """
        Moving to the destination.
        Update _elapsed_steps, localIdleness and self.pos
        Return the idleness of the current position (scalar).
        No argument.
        """
        self._elapsed_steps += 1
        targetPos = self.localTargetCoords[self.next_target_ind]  # 目的地坐标
        targetIdleness = self.localIdleness[
            self.next_target_ind
        ]  # 此刻的idleness暂时变成目标位置的idleness且暂不清零，因为有可能一步到不了
        currPos = self.pos
        dis2Target = np.linalg.norm(targetPos - currPos)

        # update local idleness
        self.localIdleness += 1.0  # localIdleness是向量 记录所有地方的idleness，此刻全部都加一
        if dis2Target == 0:  # 如果我就在目的地上 那我就不动了
            self.isOnNode = True
            #############################################
            self.pos = targetPos
            self.localIdleness[self.next_target_ind] = 0  # 此时目的地即我的位置的idleness清零
            #############################################
            return targetIdleness
        elif dis2Target <= ALPHA * self.step_length:  # 如果我离目的地很近，小于一个步长
            self.pos = targetPos  # 移动自己到目的地上 从而实现了move
            self.isOnNode = True
            self.localIdleness[self.next_target_ind] = 0
            return targetIdleness
        else:  # 没法一步到目的地，在边上
            nvec = (targetPos - currPos) / dis2Target  # 单位向量，方向是从现在的位置指向target
            self.pos = (
                currPos + self.step_length * nvec
            )  # 此刻我的位置变成了往前挪一步后的位置，即实现了move
            self.isOnNode = False
            return -1

    def randomPolicy(self, allow_reverse=True):
        """
        randomly select the next target index

        """

        if self.isOnNode:
            curr_ind = findNodeIndexByPos(self.localMap, self.pos)
            nbr_ind = list(self.localMap.neighbors(curr_ind))
            return np.random.choice(nbr_ind)
        else:
            edge = findEdgeByPos(
                self.localMap, self.pos, self.next_target_ind, self.localTargetCoords
            )
            if allow_reverse:
                return np.random.choice(edge)
            else:
                return self.next_target_ind

    def greedyPolicy(self, allow_reverse=True):
        """
        greedily select the next target index with the highest idleness value.

        """

        if self.isOnNode:
            curr_ind = findNodeIndexByPos(self.localMap, self.pos)
            nbr_ind = list(self.localMap.neighbors(curr_ind))
            max_ind = np.argmax(self.localIdleness[nbr_ind])
            return nbr_ind[max_ind]
        else:
            edge = findEdgeByPos(
                self.localMap, self.pos, self.next_target_ind, self.localTargetCoords
            )
            if allow_reverse:
                max_ind = np.argmax(self.localIdleness[edge])
                return edge[max_ind]
            else:
                return self.next_target_ind

    def __str__(self):
        return f"Agent-{self.ID}, pos: {self.pos}, prev_target_ind: {self.prev_target_ind}, next_target_ind: {self.next_target_ind}"


class MAPatrolEnv(gym.Env):
    # def __init__(self, numAgents_range=(4, 4), numTargets_range=(20, 20), env_size=(1.0, 1.0), min_dist=None, nn_range=(3, 5), comms_radius=None):
    def __init__(
        self,
        numAgents=4,
        numTargets_range=(10, 10),
        env_size=(1.0, 1.0),
        min_dist=None,
        nn_range=(3, 5),
        comms_radius=None,
    ):
        """
        :param numAgents_range: range of number of agents
        :param numTargets_range: range of number of targets (POIs)
        :param env_size: a tuple representing (height, width)
        :param nn_range: range of number of nearest neighboring targets
        :param comms_radius: communication range radius, None if global communication
        """
        super().__init__()
        # self.numAgents_range = numAgents_range
        self.numTargets_range = numTargets_range
        self.numAgents = numAgents
        self.env_size = env_size
        self.min_dist = min(self.env_size) / 20.0 if min_dist is None else min_dist
        self.nn_range = nn_range
        self.comms_radius = comms_radius
        self.comms_rad_square = self.comms_radius**2 if self.comms_radius is not None else None
        self.targetCoords = None
        self.numTargets = None
        self.map = None
        self.agents = None
        self.fig = None
        self.comms_circle = None
        self._elapsed_steps = None
        self._max_episode_steps = None
        self.frames = []
        self.dim_connectivity = None
        self.connectivity_feature = None

    ############################
    def params_from_cfg(self, cfgs):
        """
        Overwrite env parameters if additional configs provided
        Default params. params.cfg will cover this, get the value from here if params.cfg doesn't contain certain para.
        """
        self.numAgents_range = (cfgs.getint("numAgents0"), cfgs.getint("numAgents1"))
        self.numTargets_range = (cfgs.getint("numTargets0"), cfgs.getint("numTargets1"))
        self.nn_range = (cfgs.getint("numNeighbors0"), cfgs.getint("numNeighbors1"))
        self.env_size = (cfgs.getfloat("env_size0"), cfgs.getfloat("env_size1"))
        self._max_episode_steps = cfgs.getint("max_episode_steps")
        self.dim_connectivity = cfgs.getint("dim_cf")

    ############################

    def _initEnv(self):
        self.numTargets = np.random.randint(self.numTargets_range[0], self.numTargets_range[1] + 1)
        ##############
        # self.numAgents = np.random.randint(self.numAgents_range[0], self.numAgents_range[1]+1)
        ##############
        self.targetCoords = generate_points_with_min_distance(
            self.numTargets, self.env_size, self.min_dist
        )
        self.map = self._buildMap()
        self.globalIdleness = np.zeros(self.numTargets)
        self.connectivity_feature = self.get_connectivity_feature(self.dim_connectivity)

    def _buildMap(self, weighted=True):
        X = self.targetCoords
        k = self.nn_range[1] + 1
        neigh = NearestNeighbors(n_neighbors=k)
        neigh.fit(X)
        neigh_dist, neigh_ind = neigh.kneighbors(X)
        G = nx.Graph()  # undirected graph

        # add nodes; use two for loops to keep the order of nodes the same
        for i in range(self.numTargets):
            G.add_node(i, pos=X[i])  # add node i
        # add edges while ignoring self-loops
        for i in range(self.numTargets):
            num_nbr = np.random.randint(self.nn_range[0], self.nn_range[1] + 1)
            for j in range(1, num_nbr + 1):
                G.add_edge(
                    i, neigh_ind[i, j], weight=neigh_dist[i, j]
                )  # add node i, neigh_ind[i,j] and connect these two nodes, add distance as weight

        eig_vals = nx.laplacian_spectrum(G)
        second_smallest_eig_val = eig_vals[1]

        # ensure the map to be connected
        while second_smallest_eig_val <= 0:  # Not connected
            G.clear_edges()
            for i in range(self.numTargets):
                num_nbr = np.random.randint(self.nn_range[0], self.nn_range[1] + 1)
                for j in range(1, num_nbr + 1):
                    if weighted:
                        G.add_edge(i, neigh_ind[i, j], weight=neigh_dist[i, j])
                    else:
                        G.add_edge(i, neigh_ind[i, j])

            eig_vals = nx.laplacian_spectrum(G)
            second_smallest_eig_val = eig_vals[1]
        return G

    def get_connectivity_feature(self, dim_keep):
        """
        Args
            dim_keep: keep first dim_keep columns of the eigen vectors
        return
            first dim_keep columns of the eigen vectors
        """
        G = self.map
        # A = nx.adjacency_matrix(G,nodelist=G.nodes())
        A = nx.adjacency_matrix(G).todense()
        eigenvalues, eigenvectors = np.linalg.eig(A)  # eigenvalues: numpy.array (num_target, )
        # eigenvectors: matrix (num_target, num_target) matrix([[], ..., [], []])
        # vec[:,i] is the eigenvector corresponding to the eigenvalue value[i].
        # ind = eigenvalues.argsort()[::-1]
        ind = np.argsort(eigenvalues)
        eigenvec_sorted = eigenvectors[:, ind].A
        connectivity_feature = eigenvec_sorted[:, :dim_keep]
        connectivity_feature = np.expand_dims(connectivity_feature, axis=0)
        return connectivity_feature

    def communicate(self, ID):
        """
        Clear previous neigbors information (localIdleness Map and Obs)
        Update cachedLocalIdleness and cachedNeighborObs to current neigbors information (localIdleness Map and Obs)
        """
        self.agents[ID - 1].cachedLocalIdleness.clear()  # 清空之前邻居的localIdleness
        self.agents[ID - 1].cachedNeighborObs.clear()  # 清空之前邻居的obs
        agentPos = np.array([a.pos for a in self.agents]).reshape(
            -1, 2
        )  # 所有agent的位置包括自己 shape(num_agents,2)
        ownPos = self.agents[ID - 1].pos.reshape(1, 2)  # 我自己的position, shape(1,2)
        relDiff = ownPos - agentPos  # 所有agent和我的坐标差 shape(num_agents,2)
        relDisSqaure = (
            relDiff[:, 0] ** 2 + relDiff[:, 1] ** 2
        )  # shape(num_agents,) [0.17,0,0.30,0.40] ID = 2, 所以第1列为0
        nbr_ind = (
            relDisSqaure.argsort()
        )  # 如[0.17,0,0.30,0.40] -> [1,0,2,3] 按照距离从小到大排列，不是按照连不连接

        # limited communication range
        if self.comms_radius is not None:
            # excluding myself
            for nbr in nbr_ind[1:]:  # 不包括自己的所有邻居
                if relDisSqaure[nbr] <= self.comms_rad_square:  # 如果在通讯范围内
                    self.agents[ID - 1].cachedLocalIdleness.append(
                        self.agents[nbr].localIdleness
                    )  # 把每个邻居的idlenssmap都append起来
                    nbrObs = (
                        self.agents[nbr].pos,
                        self.agents[ID - 1].localTargetCoords[self.agents[nbr].next_target_ind],
                    )  # agent state! ([该邻居的坐标], [邻居接下来要去地点的坐标])
                    self.agents[ID - 1].cachedNeighborObs.append(
                        nbrObs
                    )  # 把每个邻居的Obs也append起来

        # global communication
        else:  # 设置comms_radius为none即为全局通信
            for nbr in nbr_ind[1:]:
                self.agents[ID - 1].cachedLocalIdleness.append(self.agents[nbr].localIdleness)
                nbrObs = (
                    self.agents[nbr].pos,
                    self.agents[ID - 1].localTargetCoords[self.agents[nbr].next_target_ind],
                )
                self.agents[ID - 1].cachedNeighborObs.append(
                    nbrObs
                )  # 这里更新agent里的self.cachedNeighborObs属性

    def observe(self, ID, relative=True):
        """
        In the view of agent No. ID, get the target states and agent states.
        """
        localIdleness = self.agents[ID - 1].fuse()  # array, shape(numTargets, )
        agent_states = self.agents[ID - 1].getNbrObs(relative)
        agent_states = np.expand_dims(agent_states, axis=0)
        target_coords = self.agents[ID - 1].get_target_coords(
            relative
        )  # array, shape(numTargets, 2)
        target_states = np.hstack(
            (target_coords, localIdleness.reshape(-1, 1))
        )  # array, shape(numTargets, 3)
        target_states = np.expand_dims(target_states, axis=0)
        connectivity_feature = self.connectivity_feature

        ############## update GLOBAL idleness and return local mask
        local_A = nx.adjacency_matrix(
            self.agents[ID - 1].localMap, weight=None
        ).todense()  # Adj Max. shape(num_target,num_target)
        if self.agents[ID - 1].isOnNode:  # 如果这个agent在点上
            target_ind = findNodeIndexByPos(
                self.agents[ID - 1].localMap, self.agents[ID - 1].pos
            )  # 找出此刻agent在哪个索引的点上 localMap就是envMap
            self.globalIdleness[target_ind] = (
                0  # globalIdleness shape: (num_target,) [0,0,...,0] 把该点的idleness置零
            )
            localMask = np.array(
                local_A[target_ind] == 1
            )  # LocalMask记录了哪些点与target_ind这个点连接
        else:
            edge = findEdgeByPos(
                self.agents[ID - 1].localMap,
                self.agents[ID - 1].pos,
                self.agents[ID - 1].next_target_ind,
                self.agents[ID - 1].localTargetCoords,
            )  # [ind1, ind2]
            localMask = np.zeros(self.numTargets, dtype=bool)
            localMask[edge] = 1
            localMask = np.expand_dims(localMask, axis=0)
        ##############
        localMask = np.expand_dims(localMask, axis=0)
        return agent_states, target_states, connectivity_feature, localMask

    def tick(self):
        self._elapsed_steps += 1

    def checkFinished(self):
        """
        Check if all agents are finished.
        """
        return True if np.all([a.finished for a in self.agents]) else False

    def step(self, ID, next_target_ind, relative=True, localReward=True):
        """
        对于指定ID的agent 通过NN计算出的下一步去的节点编号(即next_target_ind) 实施该action 更新地图idleness
        返回 Target States: relative coord, localIdleness
            Agent States: relative coord, relative coord of the next node to visit, localMask, finished, reward
        """
        self.agents[ID - 1].next_target_ind = next_target_ind
        tmpRwrd = self.agents[
            ID - 1
        ].move()  # 更新agent此时位置的reward指标即idleness（不是reward）,若在点上返回idleness,在边上返回-1
        agentStates, targetStates, connectivity_feature, localMask = self.observe(ID, relative)

        # check if agent has finished
        if self.agents[ID - 1].elapsed_steps >= self.max_episode_steps:
            self.agents[ID - 1].finished = True

        if localReward:  # 环境如何给reward
            if tmpRwrd < 0:  # tmpRwrd表示在不在边上，在边上为-1即小于0
                reward = -0.1
            else:
                reward = tmpRwrd / self.max_episode_steps
            return (
                agentStates,
                targetStates,
                connectivity_feature,
                localMask,
                self.agents[ID - 1].localTargetCoords,
                self.agents[ID - 1].finished,
                reward,
            )
        else:  ###这个是啥意思，一直没用到？
            return (
                agentStates,
                targetStates,
                connectivity_feature,
                localMask,
                self.agents[ID - 1].localTargetCoords,
                self.agents[ID - 1].finished,
            )  # localMask??

    @property
    def max_episode_steps(self):
        return self._max_episode_steps

    @property
    def elapsed_steps(self):
        return self._elapsed_steps

    def updateEnv(self, localReward=True):
        """
        更新env的global Idleness属性
        """
        self.tick()
        occupied_ind = list(range(self.numTargets))
        for a in self.agents:
            if a.isOnNode:
                ind = findNodeIndexByPos(self.map, a.pos)
                if ind in occupied_ind:
                    occupied_ind.remove(ind)
        self.globalIdleness[occupied_ind] += 1.0
        if not localReward:
            return -np.mean(self.globalIdleness) / self.max_episode_steps

    # def reset(self, scattered=False, max_episode_steps=10):
    def reset(self, scattered=False):
        # self._max_episode_steps = max_episode_steps # works for both wrapped env and unwrapped env
        self._elapsed_steps = 0
        self.frames.clear()
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
        self._initEnv()
        # self.numAgents = np.random.randint(self.numAgents_range[0], self.numAgents_range[1]+1)
        if not scattered:
            depot_ind = np.random.choice(self.numTargets)
            self.agents = [
                Agent(
                    ID,
                    self.targetCoords[depot_ind],
                    self.map,
                    self.targetCoords,
                    np.copy(self.globalIdleness),
                )
                for ID in range(1, self.numAgents + 1)
            ]
        else:
            pos_ind = np.random.choice(self.numTargets, size=self.numAgents)
            self.agents = [
                Agent(
                    ID,
                    self.targetCoords[pos_ind[ID - 1]],
                    self.map,
                    self.targetCoords,
                    np.copy(self.globalIdleness),
                )
                for ID in range(1, self.numAgents + 1)
            ]

    def render(self, save=False, show_comms=False):
        if self.fig is None:
            plt.ion()  # turn on interactive mode
            self.fig, self.ax = plt.subplots(figsize=(8, 8))
            self.ax.set_xlim([-0.1 * self.env_size[1], 1.1 * self.env_size[1]])
            self.ax.set_ylim([-0.1 * self.env_size[0], 1.1 * self.env_size[0]])

            # plot targets
            self.targets_sc = self.ax.scatter(
                self.targetCoords[:, 0], self.targetCoords[:, 1], marker="s", s=10**2
            )
            self.targets_sc.set_color([1.0, 0.0, 0.0, 0.0])
            self.targets_sc.set_edgecolor([0.0, 0.0, 0.0, 1.0])

            # annotate target indexes
            for i, pos in enumerate(self.targetCoords):
                self.ax.text(
                    pos[0] + self.env_size[1] / 100.0, pos[1] + self.env_size[0] / 100.0, str(i)
                )
            for e in self.map.edges():
                self.ax.plot(
                    *zip(self.targetCoords[e[0]], self.targetCoords[e[1]]), c="k", lw=0.5, ls="-."
                )

            # plot agents
            self.agents_sc = self.ax.scatter([], [], c="b", marker="^", s=6**2)
            if show_comms and self.comms_radius is not None:
                if self.comms_circle is None:
                    numPlotPoints = int(self.comms_radius / min(self.env_size) * 100)
                    self.comms_circle = [
                        self.comms_radius * np.cos(np.linspace(0, 2 * np.pi, numPlotPoints)),
                        self.comms_radius * np.sin(np.linspace(0, 2 * np.pi, numPlotPoints)),
                    ]
                self.agents_comms_circle_l = []
                for a in self.agents:
                    (comms_circle_l,) = self.ax.plot(
                        *map(lambda x: sum(x), zip(a.pos, self.comms_circle)), c="g", ls="--", lw=1
                    )
                    self.agents_comms_circle_l.append(comms_circle_l)

        # update agent pos
        agent_x = [a.pos[0] for a in self.agents]
        agent_y = [a.pos[1] for a in self.agents]
        self.agents_sc.set_offsets(np.c_[agent_x, agent_y])
        if show_comms and self.comms_radius is not None:
            for i, comms_circle_l in enumerate(self.agents_comms_circle_l):
                comms_circle = list(
                    map(lambda x: sum(x), zip(self.agents[i].pos, self.comms_circle))
                )
                comms_circle_l.set_xdata(comms_circle[0])
                comms_circle_l.set_ydata(comms_circle[1])

        # update idleness
        global_idleness_c = [
            [1.0, 0.0, 0.0, idleness / self.max_episode_steps] for idleness in self.globalIdleness
        ]  # different shade of red according to the idleness
        self.targets_sc.set_color(global_idleness_c)
        self.targets_sc.set_edgecolor([0.0, 0.0, 0.0, 1.0])
        # display
        self.ax.set_title(
            f"[step {self.elapsed_steps}|{self.max_episode_steps}] global average idleness: {round(self.globalIdleness.mean(),2)}, global worst idleness: {round(self.globalIdleness.max(),1)}"
        )

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        if save:
            self.frames.append(grab_rgb_frame(self.fig.canvas))
        return self.globalIdleness.mean(), self.globalIdleness.max()

    def save(self, filename, fps=10, loop=1):
        if self.frames:
            imageio.mimwrite(filename, self.frames, fps=fps, loop=loop)
