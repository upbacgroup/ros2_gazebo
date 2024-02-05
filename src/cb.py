#!/usr/bin/env python3
import rospy
import gym
from gym.spaces import Box

import stable_baselines3 as sb3
from stable_baselines3.common.env_wrappers import NormalizeObservations, FlattenObservations
from stable_baselines3.common.noise import NormalActionNoise

from gazebo_ros_pkgs.msg import ModelStates


class DroneTargetCollectionEnv(gym.Env):
    def __init__(self):
        self.model_states_topic = rospy.get_param("~model_states_topic")
        self.target_points_topic = rospy.get_param("~target_points_topic")

        self.drone_state_sub = rospy.Subscriber(self.model_states_topic, ModelStates, self.drone_state_cb)
        self.target_points_sub = rospy.Subscriber(self.target_points_topic, Object, self.target_points_cb)

        self.action_space = Box(-1, 1, shape=(4,))
        self.observation_space = Box(-1, 1, shape=(6,))

        self.done = False
        self.collected_target_points = 0

    def drone_state_cb(self, msg):
        self.drone_state = msg.pose[0]

    def target_points_cb(self, msg):
        self.target_points = msg.position

    def step(self, action):
        self.take_action(action)

        if self.is_done():
            reward = 0
        else:
            reward = -(self.collected_target_points / self.max_target_points)

        self.done = self.is_done()

        return self.get_observation(), reward, self.done, {}

    def get_observation(self):
        observation = [
            self.drone_state.position.x,
            self.drone_state.position.y,
            self.drone_state.orientation.w,
            self.distance_to_closest_target_point(),
            self.angle_to_closest_target_point(),
        ]

        return observation

    def is_done(self):
        if self.collected_target_points == self.max_target_points:
            return True
        else:
            return False

    def distance_to_closest_target_point(self):
        closest_distance = float("inf")

        for target_point in self.target_points:
            distance = self.drone_state.position.distance(target_point)
            if distance < closest_distance:
                closest_distance = distance

        return closest_distance

    def angle_to_closest_target_point(self):
        closest_target_point = self.target_points[0]
        closest_angle = -1.0

        for target_point in self.target_points:
            angle = self.drone_state.position.angle_to(target_point)
            if angle > closest_angle:
                closest_angle = angle

        if closest_distance == 0:
            closest_angle = 0

        return closest_angle

    def take_action(self, action):
        raise NotImplementedError("Take action method not implemented")


def main():
    rospy.init_node("drone_target_collection_env")

    env = DroneTargetCollectionEnv()
    model = sb3.PPO("MlpPolicy", env)

    model.learn(total_timesteps=100000)

    while not rospy.is_shutdown():
        action = model.predict(env.get_observation())[0]
        _, _, done, _ = env.step(action)

        if done:
            print("Done, collected", env.collected_target_points, "target points")
            env.reset()


if __name__ == "__main__":
    main()
