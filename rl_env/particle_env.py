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
from stochastic.processes.continuous import BrownianMotion
alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


class raw_env(SimpleEnv, EzPickle):
    def __init__(
        self,
        num_agents=10,
        num_food_sources=1,
        obs_type = 'proprioception',
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
        world = scenario.make_world(num_agents, num_food_sources, obs_type, flow)
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
    
    # override reset function to fit pettingzoo/gym standard of returning the initial observations
    def reset(self, seed=None, options=None):
        if seed is not None:
            self._seed(seed=seed)
        observations, info = self.scenario.reset_world(self.world, self.np_random)

        self.agents = self.possible_agents[:]
        self.rewards = {name: 0.0 for name in self.agents}
        self._cumulative_rewards = {name: 0.0 for name in self.agents}
        self.terminations = {name: False for name in self.agents}
        self.truncations = {name: False for name in self.agents}
        self.infos = {name: {} for name in self.agents}

        self.agent_selection = self._agent_selector.reset()
        self.steps = 0

        self.current_actions = [None] * self.num_agents

        return observations
    
    # override draw function from MPE2 package 
    def draw(self):
        # clear screen
        self.screen.fill((255, 255, 255))

        # flip entities so landmakrs are drawn last (they will be rendered in the background)
        entities_flipped = self.world.entities[::-1]

        # update bounds to center around agent
        #all_poses = [entity.state.p_pos for entity in entities_flipped]

        # We want to prohibit agents moving outside the screen, so we choose a fixed cam_range
        cam_range = self.original_cam_range

        # The scaling factor is used for dynamic rescaling of the rendering - a.k.a Zoom In/Zoom Out effect
        # The 0.9 is a factor to keep the entities from appearing "too" out-of-bounds
        scaling_factor = 0.9 * cam_range

        # update geometry and text positions
        text_line = 0
        for e, entity in enumerate(entities_flipped):
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
        #print("step is done")
        #print(" ")
        #print(" ")
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

env = make_env(raw_env) # this throws an error for the return of the overwritten reset function (only for "env", not for "raw-env" or "parallel_env")
#env = raw_env
parallel_env = parallel_wrapper_fn(env)

class Scenario(BaseScenario):
    def make_world(self, num_agents=10, num_food_sources=1, obs_type = 'proprioception', flow='none'):
        world = MyWorld()
        num_agents = num_agents
        num_food_sources = num_food_sources
        print(obs_type)
        world.obs_type = obs_type
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = f"agent_{i}"
            agent.collide = True # they will be propulsed away if they collide -> check, why this happens at the t+1 right now!
            agent.size = 0.02
            agent.silent = True # we don't work with communication in this env
            # we want the agents to be affected by Brownian motion
            # (Implemented in MyWorld class below)
            agent.u_noise = True
            agent.boundary = False
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_food_sources)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = "food source %d" % i
            landmark.collide = False
            landmark.movable = False
            landmark.size = 0.2
            landmark.boundary = False # If True, landmark cannot be seen in observations
        return world

    def reset_world(self, world, np_random):
        # greenish colour for agents
        colors = [np.array([0.0, 0.0, 0.0]),
                    np.array([0.95, 0.0, 0.35]),
                    np.array([0.35, 0.35, 0.35]), 
                    np.array([0.5, 0.0, 0.35]), 
                    np.array([0.9, 0.85, 0.2]),
                    np.array([0.2, 0.85, 0.9]),
                    np.array([0.0, 0.45, 0.05]),
                    np.array([0.1, 0.85, 0.35]),
                    np.array([0.2, 0.0, 0.75]),
                    np.array([1.0, 1.0, 1.0])]
        for i, agent in enumerate(world.agents):
            #agent.color = np.array([0.35, 0.85, 0.35])
            agent.color = colors[i]
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

        observations = {}
        info = {}

        for agent in world.agents:
            observations[agent.name] = self.observation(agent, world)
            info[agent.name] = {"reset complete"}  # Replace with actual info if needed
        return observations, info

    # Function from simple spread environment to check for collisions
    def is_collision(self, entity1, entity2):
        delta_pos = entity1.state.p_pos - entity2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = entity1.size + entity2.size
        return True if dist < dist_min else False

    # Reward is given based on (1) minimizing distance to food, 
    # (2) no collisions with other agents
    # (3) staying within the environment bounds
    def reward(self, agent, world):
        dist_to_food = np.sum(np.square(agent.state.p_pos - world.landmarks[0].state.p_pos))
        penalty_for_colliding = 0.0
        if agent.collide:
            for a in world.agents:
                penalty_for_colliding -= 1.0 * (self.is_collision(a, agent) and a != agent) # reward of -1 for each collision
        border_position_penalty = 0.0
        distance_to_border = 1 - np.max(np.abs(agent.state.p_pos)) # distance to the closest border
        if distance_to_border <= agent.size *1.5: # if the agent is closer to the border than 1.5 times its radius
            border_position_penalty = -1.0
        return -dist_to_food #+ border_position_penalty + penalty_for_colliding # Add after size problems and collision bug are fixed 

    def reward_new(self, agent, world):
        # Food reward
        food_reward = 0.0
        for landmark in world.landmarks:
            if self.is_collision(agent, landmark):
                food_reward += 1.0
        # Penalty for colliding
        penalty_for_colliding = 0.0
        if agent.collide:
            for a in world.agents:
                penalty_for_colliding -= 2.0 * (self.is_collision(a, agent) and a != agent) # reward of -1 for each collision
        # Penalty for trying to move outside the frame
        border_position_penalty = 0.0
        distance_to_border = 1 - np.max(np.abs(agent.state.p_pos)) # distance to the closest border
        if distance_to_border <= agent.size *1.5: # if the agent is closer to the border than 1.5 times its radius
            border_position_penalty = -1.0
        return food_reward + border_position_penalty + penalty_for_colliding # Add after size problems and collision bug are fixed 

    def observation_old(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        return np.concatenate([agent.state.p_vel] + entity_pos)

    def observation(self, agent, world):
        # get distances to all other entities for this agent and combine with its velocity
        entity_pos = []
        
        if world.obs_type == "god":
            for entity in world.entities:
                if entity.boundary:  # invisible entities
                    entity_pos.append(np.array([0.0,0.0]))
                else:
                    entity_pos.append(entity.state.p_pos) # entity_pos are the vectors to the other entities
        elif world.obs_type == "vision":
            for entity in world.entities:
                entity_pos.append(entity.state.p_pos - agent.state.p_pos) # old structure right now (only swapped landmarks for entities but this does not seem to work at all)
        elif world.obs_type == "proprioception":
            for entity in world.entities:
                if entity == agent:
                    entity_pos.append(entity.state.p_pos) # only own position added
                else:
                    entity_pos.append(np.array([0.0,0.0]))
        elif world.obs_type == "old":
            for entity in world.landmarks:
                entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        else:
            print("Incorrect observation space type. Enter \"god\", \"vision\" or \"proprioception\"")

        return np.concatenate([agent.state.p_vel] + entity_pos) # adds vel at the front of the list (no math. operations)

class MyWorld(World):
    # Override apply_action_force function from _mpe2_utils/core.py
    def apply_action_force(self, p_force):
        # set applied forces
        for i, agent in enumerate(self.agents):
            if agent.movable:
                # a Brownian step from 0.0 origin
                brownian_step = [BrownianMotion(drift=0, scale=1, t=1, rng=None).sample_at([1])[0], # if not in 2D environment, you have to adapt dimensions!
                                 BrownianMotion(drift=0, scale=1, t=1, rng=None).sample_at([1])[0]]
                noise = (
                    np.asarray(brownian_step)
                    if agent.u_noise # if set to True, otherwise no Brownian motion
                    else 0.0
                    )
                p_force[i] = agent.action.u + noise # Brownian vector is added to the action vector
        return p_force
    
    # Override step function from _mpe2/utils/core.py
    # update state of the world
    def step(self):
        # set actions for scripted agents
        for agent in self.scripted_agents:
            agent.action = agent.action_callback(agent, self)
        # gather forces applied to entities
        p_force = [None] * len(self.entities)
        # apply agent physical controls
        p_force = self.apply_action_force(p_force)

        # apply environment forces
        '''
        collision_detected = True
        i = 0
        while collision_detected:
            print("while loop started")
            p_force = self.apply_environment_force(p_force)
            # integrate physical state
            self.integrate_state(p_force)
            # update agent state
            for agent in self.agents:
                self.update_agent_state(agent)
            if self.is_any_collision():
                collision_detected = True
                print("Collision detected!")
            else: 
                False
            print("while loop ended")
            i +=1
            if i > 10:
                collision_detected = False
            print('------------------')
        '''
        p_force = self.apply_environment_force(p_force)
        # integrate physical state
        self.integrate_state(p_force)
        # update agent state
        for agent in self.agents:
            self.update_agent_state(agent)

    # Override apply_environment_force function from _mpe2_utils/core.py to account for multiple collision (new collision after corrected course)
    # gather physical forces acting on entities
    def apply_environment_force(self, p_force): 
        for a, entity_a in enumerate(self.entities):
            for b, entity_b in enumerate(self.entities):
                if b <= a:
                    continue
                [f_a, f_b] = self.get_collision_force(entity_a, entity_b)
                if f_a is not None or f_b is not None:
                    if p_force[a] is None:
                        p_force[a] = 0.0
                    p_force[a] = f_a + p_force[a]
                    if p_force[b] is None:
                        p_force[b] = 0.0
                    p_force[b] = f_b + p_force[b]
        return p_force
    
    # Override integrate_state function from _mpe2_utils/core.py
    # integrate physical state
    def integrate_state(self, p_force):
        #print('new_round')
        #print('#################')
        for i, entity in enumerate(self.entities):
            if not entity.movable:
                continue
            #entity.state.p_pos += entity.state.p_vel * self.dt
            #print('Old pos: ', entity.state.p_pos)
            entity.state.p_vel = entity.state.p_vel * (1 - self.damping) # damping is loss of energy, here 0.25
            if p_force[i] is not None:
                entity.state.p_vel += (p_force[i] / entity.mass) * self.dt # mass is 1 and dt is 0.1 (time step interval)
            if entity.max_speed is not None:
                speed = np.sqrt(
                    np.square(entity.state.p_vel[0]) + np.square(entity.state.p_vel[1])
                )
                if speed > entity.max_speed:
                    entity.state.p_vel = (
                        entity.state.p_vel
                        / np.sqrt(
                            np.square(entity.state.p_vel[0])
                            + np.square(entity.state.p_vel[1])
                        )
                        * entity.max_speed
                    )
            entity.state.p_pos += entity.state.p_vel * self.dt
            #print('New pos: ', entity.state.p_pos)
            #print('--------------------')
    
    # Override get_collision_force function from _mpe2_utils/core.py (to prevent from dividing by zero)
    # get collision forces for any contact between two entities
    def get_collision_force(self, entity_a, entity_b):
        if (not entity_a.collide) or (not entity_b.collide):
            return [None, None]  # not a collider
        if entity_a is entity_b:
            return [None, None]  # don't collide against itself
        # compute actual distance between entities
        delta_pos = entity_a.state.p_pos - entity_b.state.p_pos
        # compute Euclidean distance
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        # minimum allowable distance
        dist_min = entity_a.size + entity_b.size
        if dist >= dist_min: # agents are far enough away from each other
            return [None, None]
        # softmax penetration
        k = self.contact_margin
        penetration = np.logaddexp(0, -(dist - dist_min) / k) * k
        if dist == 0.0: # to prevent division by zero
            dist = 0.0001
        force = self.contact_force * delta_pos / dist * penetration
        force_a = +force if entity_a.movable else None
        force_b = -force if entity_b.movable else None
        return [force_a, force_b]
    
    # Function from simple spread environment to check for collisions
    def is_any_collision(self):
        collision = False
        for a, agent_a in enumerate(self.entities):
            for b, agent_b in enumerate(self.entities):
                if b <= a:
                    continue
                delta_pos = agent_a.state.p_pos - agent_b.state.p_pos
                dist = np.sqrt(np.sum(np.square(delta_pos)))
                dist_min = agent_a.size + agent_b.size
                if dist < dist_min:
                    collision = True
        return collision