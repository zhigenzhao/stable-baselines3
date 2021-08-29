import os
import numpy as np

import pinocchio
import pybullet
import pybullet_data
from pybullet_utils import bullet_client

import gym


class KukaPlanarEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, q_goal, N=200, renders=True):
        super(KukaPlanarEnv, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        # KUKA_TORQUE_UB = np.array([320, 320, 176, 176, 110, 40, 40])
        KUKA_TORQUE_UB = np.array([320, 176, 40])
        self.action_space = gym.spaces.Box(low=-KUKA_TORQUE_UB, high=KUKA_TORQUE_UB)
        # Example for using image as input:
        self.observation_space = gym.spaces.Box(low=-float("inf"), high=float("inf"), shape=(6,))
        print(self.observation_space)

        self._renders = renders
        self._physics_client_id = -1

        self._p = None
        self.state = None
        self.kuka = None
        self.dt = 1e-2
        self.N = N
        self.n_step = 0
        self.iiwa_urdf = "/home/zzhao300/code/manipulation-learning/examples/kuka_planar/kuka_models/model_planar.urdf"

        self.active_joint_idx = [1, 3, 5]
        self.q_goal = q_goal
        self.ee_desired = None
        self.ee_current = None

        self.iiwa_model = pinocchio.buildModelFromUrdf(self.iiwa_urdf)
        self.iiwa_data = self.iiwa_model.createData()

    def step(self, action):
        # Execute one time step within the environment
        q, vq = self._pinocchio_fd(action)

        # self._p.setJointMotorControlArray(
        #     bodyUniqueId = self.kuka,
        #     jointIndices = self.active_joint_idx,
        #     controlMode = pybullet.POSITION_CONTROL,
        #     targetPositions = q,
        #     targetVelocities = vq
        # )

        # self._p.stepSimulation()
        num_active_joints = len(self.active_joint_idx)
        for i in range(num_active_joints):
            self._p.resetJointState(
                self.kuka, self.active_joint_idx[i], q[i], vq[i])

        self.n_step += 1
        link_state = self._p.getLinkState(
            bodyUniqueId = self.kuka,
            linkIndex = 6,
            computeForwardKinematics=True
        )

        self.ee_current = np.array(link_state[0])

        self.state = self._get_state()

        done = self.n_step >= self.N

        reward = self._kuka_planar_cost(
            np.array(self.state),
            np.array(action).reshape((-1, 1)),
            self.q_goal
        )

        return self.state, -reward, done, {}

    def reset(self, x0=None):
        self.n_step = 0
        # Reset the state of the environment to an initial state
        if self._physics_client_id < 0:
            if self._renders:
                self._p = bullet_client.BulletClient(pybullet.GUI)
                self._p.resetDebugVisualizerCamera(
                    cameraDistance=2.5,
                    cameraYaw=30,
                    cameraPitch=-45,
                    cameraTargetPosition=[0,0,0.25])
            else:
                self._p = bullet_client.BulletClient()

            self._physics_client_id = self._p._client
            self._p.resetSimulation()
            self._p.setAdditionalSearchPath(pybullet_data.getDataPath())
            self.ground = self._p.loadURDF("plane.urdf")
            self.kuka = self._p.loadURDF(self.iiwa_urdf, [0, 0, 0], useFixedBase=1)
            self.num_joints = self._p.getNumJoints(self.kuka)
            # for i in range(num_joints):
            #     print(self._p.getJointInfo(self.kuka, i)[1])

            self._p.setGravity(0, 0, -9.81)
            self._p.setTimeStep(self.dt)
            self._p.setRealTimeSimulation(0)

        x_goal = np.zeros(self.num_joints*2)
        for i in range(len(self.active_joint_idx)):
            x_goal[self.active_joint_idx[i]] = self.q_goal[i]

        for i in range(self.num_joints):
            self._p.resetJointState(self.kuka, i, x_goal[i], x_goal[i+7])

        # FK to compute "desired EE coordinate"
        link_state = self._p.getLinkState(
            bodyUniqueId = self.kuka,
            linkIndex = 6,
            computeForwardKinematics=True
        )

        self.ee_desired = np.array(link_state[0])

        state = np.zeros(self.num_joints*2)
        if x0 is None:
            x0 = self._sample_kuka_planar_pos()
        for i in range(len(self.active_joint_idx)):
            state[self.active_joint_idx[i]]=x0[i]

        for i in range(self.num_joints):
            self._p.resetJointState(self.kuka, i, state[i], state[i+7])

        self.state = self._get_state()

        return self.state

    def _get_state(self):
        pos = []
        vel = []
        for i in self.active_joint_idx:
            pos.append(self._p.getJointState(self.kuka, i)[0])
            vel.append(self._p.getJointState(self.kuka, i)[1])

        state = pos+vel
        return state

    def _sample_kuka_planar_pos(self, sigma=np.pi):
        rnd = (np.random.random_sample((3,))-0.5)*2 # [-1, 1)
        return rnd*sigma

    def _pinocchio_fd(self, action):
        num_active_joints = len(self.active_joint_idx)
        q = np.array(self.state[:num_active_joints]).reshape((-1, 1))
        vq = np.array(self.state[num_active_joints:]).reshape((-1, 1))
        aq0 = np.zeros_like(q)

        b = pinocchio.rnea(self.iiwa_model, self.iiwa_data, q, vq, aq0).reshape((-1, 1))
        M = pinocchio.crba(self.iiwa_model, self.iiwa_data, q)

        tau = np.array(action).reshape((-1, 1))
        aq = np.matmul(np.linalg.inv(M), (tau - b))
        vq += aq*self.dt

        q = pinocchio.integrate(self.iiwa_model, q, vq*self.dt)

        return q, vq

    def _kuka_planar_cost(self, x, u, q_goal):
        """
            l(x, u) = (x-x_ref)'W(x-x_ref)
        """
        x_ref = np.append(q_goal, [0]*3)
        W = np.array([1000]*6)
        goal_tracking_cost = np.dot(W, (x-x_ref)**2).item()

        Q = np.array([1e-2]*3+[1e-1]*3)
        x_reg_cost = np.dot(Q, x**2).item()

        R = np.array([1e-2]*3)
        u_reg_cost = np.dot(R, u**2).item()

        return (goal_tracking_cost+u_reg_cost+x_reg_cost)/1e3

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        pass

    def close(self):
        if self._physics_client_id >= 0:
            self._p.disconnect()
        self._physics_client_id = -1