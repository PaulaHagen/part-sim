"""
# Particle Environment

| Import             |      `import particle_v1`      |
|--------------------|----------------------------------------|
| Actions            | Discrete/Continuous                    |
| Parallel API       | Yes                                    |
| Manual Control     | No                                     |
| Agents             | `agents= [agent_0, agent_1,...]`       |
| Agents             | 10                                     |
| Action Shape       | (5)                                    |
| Action Values      | Discrete(5)/Box(0.0, 1.0, (5,))        |
| Observation Shape  | (4)                                    |
| Observation Values | (-inf,inf)                             |
| State Shape        | (4,)                                   |
| State Values       | (-inf,inf)                             |


In this environment a single agent sees a landmark position and is rewarded based on how close it gets to the landmark (Euclidean distance). This is not a multiagent environment, and is primarily intended for debugging purposes.

Observation space: `[self_vel, landmark_rel_position]`

### Arguments

``` python
particle_v1.env(max_cycles=25, num_agents=10, num_food_sources=1, flow='none', continuous_actions = False, dynamic_rescaling=False)
```



`max_cycles`:  number of frames (a step for each agent) until game terminates

`continuous_actions`: Whether agent action spaces are discrete(default) or continuous

`num_agents`: number of agents

`num_food_sources`: number of food sources (landmarks)

`flow`: type of flow: 'none' (default), 'circular', 'random'

`dynamic_rescaling`: Whether to rescale the size of agents and landmarks based on the screen size

"""

import numpy as np
from gymnasium.utils import EzPickle
from pettingzoo.utils.conversions import parallel_wrapper_fn

from mpe2._mpe_utils.core import Agent, Landmark, World
from mpe2._mpe_utils.scenario import BaseScenario
from mpe2._mpe_utils.simple_env import SimpleEnv, make_env
import pygame
alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


class raw_env(SimpleEnv, EzPickle):
    def __init__(
        self,
        num_agents=10,
        num_food_sources=1,
        flow = 'none',
        max_cycles=25,
        continuous_actions=False,
        render_mode=None,
        dynamic_rescaling=False,
    ):
        EzPickle.__init__(
            self,
            num_agents=num_agents,
            num_food_sources=num_food_sources,
            flow=flow,
            max_cycles=max_cycles,
            continuous_actions=continuous_actions,
            render_mode=render_mode,
        )
        scenario = Scenario()
        world = scenario.make_world(num_agents, num_food_sources, flow)
        SimpleEnv.__init__(
            self,
            scenario=scenario,
            world=world,
            render_mode=render_mode,
            max_cycles=max_cycles,
            continuous_actions=continuous_actions,
            dynamic_rescaling=dynamic_rescaling,
        )
        self.metadata["name"] = "particle_v1"

    # override draw function from MPE2 package 
    def draw(self):
        # clear screen
        self.screen.fill((255, 255, 255))

        # update bounds to center around agent
        all_poses = [entity.state.p_pos for entity in self.world.entities]

        # We want to prohibit agents moving outside the screen, so we choose a fixed cam_range
        cam_range = self.original_cam_range

        # The scaling factor is used for dynamic rescaling of the rendering - a.k.a Zoom In/Zoom Out effect
        # The 0.9 is a factor to keep the entities from appearing "too" out-of-bounds
        scaling_factor = 0.9 * cam_range

        # update geometry and text positions
        text_line = 0
        for e, entity in enumerate(self.world.entities):
            # geometry
            x, y = entity.state.p_pos
            y *= (
                -1
            )  # this makes the display mimic the old pyglet setup (ie. flips image)
            x = (
                (x / cam_range) * self.width // 2 * 0.9
            )  # the .9 is just to keep entities from appearing "too" out-of-bounds
            y = (y / cam_range) * self.height // 2 * 0.9
            x += self.width // 2
            y += self.height // 2

            # 350 is an arbitrary scale factor to get pygame to render similar sizes as pyglet
            if self.dynamic_rescaling:
                radius = entity.size * 350 * scaling_factor
            else:
                radius = entity.size * 350

            pygame.draw.circle(self.screen, entity.color * 200, (x, y), radius)
            pygame.draw.circle(self.screen, (0, 0, 0), (x, y), radius, 1)  # borders
            assert (
                0 < x < self.width and 0 < y < self.height
            ), f"Coordinates {(x, y)} are out of bounds."

            # text
            if isinstance(entity, Agent):
                if entity.silent:
                    continue
                if np.all(entity.state.c == 0):
                    word = "_"
                elif self.continuous_actions:
                    word = (
                        "[" + ",".join([f"{comm:.2f}" for comm in entity.state.c]) + "]"
                    )
                else:
                    word = alphabet[np.argmax(entity.state.c)]

                message = entity.name + " sends " + word + "   "
                message_x_pos = self.width * 0.05
                message_y_pos = self.height * 0.95 - (self.height * 0.05 * text_line)
                self.game_font.render_to(
                    self.screen, (message_x_pos, message_y_pos), message, (0, 0, 0)
                )
                text_line += 1
    
    # override _execute_world_step function
    def _execute_world_step(self):
        # set action for each agent
        for i, agent in enumerate(self.world.agents):
            action = self.current_actions[i]
            scenario_action = []
            if agent.movable:
                mdim = self.world.dim_p * 2 + 1
                if self.continuous_actions:
                    scenario_action.append(action[0:mdim])
                    action = action[mdim:]
                else:
                    scenario_action.append(action % mdim)
                    action //= mdim
            if not agent.silent:
                scenario_action.append(action)
            self._set_action(scenario_action, agent, self.action_spaces[agent.name])

        self.world.step()
        # after action is chosen, clip the positions of all entities to stay in bounds
        for entity in self.world.entities:
            entity.state.p_pos = np.clip(
                entity.state.p_pos,
                -self.original_cam_range,
                self.original_cam_range
            )

        global_reward = 0.0
        if self.local_ratio is not None:
            global_reward = float(self.scenario.global_reward(self.world))

        for agent in self.world.agents:
            agent_reward = float(self.scenario.reward(agent, self.world))
            if self.local_ratio is not None:
                reward = (
                    global_reward * (1 - self.local_ratio)
                    + agent_reward * self.local_ratio
                )
            else:
                reward = agent_reward

            self.rewards[agent.name] = reward


env = make_env(raw_env)
parallel_env = parallel_wrapper_fn(env)


class Scenario(BaseScenario):
    def make_world(self, num_agents=10, num_food_sources=1, flow='none'):
        world = World()
        num_agents = num_agents
        num_food_sources = num_food_sources
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = f"agent_{i}"
            agent.collide = False # change later
            agent.size = 0.05
            agent.silent = True
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_food_sources)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = "food source %d" % i
            landmark.collide = False # change later
            landmark.movable = False
            landmark.size = 0.2
            landmark.boundary = False
        return world

    def reset_world(self, world, np_random):
        # greenish colour for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.35, 0.85, 0.35])
        # orange colour for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.95, 0.45, 0.15])
        world.landmarks[0].color = np.array([0.95, 0.45, 0.15])
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np_random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p) # change p_vel if you want continuous movement
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np_random.uniform(-0.9, +0.9, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)

    def reward(self, agent, world):
        dist2 = np.sum(np.square(agent.state.p_pos - world.landmarks[0].state.p_pos))
        return -dist2

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        return np.concatenate([agent.state.p_vel] + entity_pos) # entity_pos are the vectors to the other entities
