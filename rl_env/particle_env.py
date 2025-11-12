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

`num_agents`: number of agents

`num_food_sources`: number of food sources (landmarks)

`flow`: type of flow: 'none' (default), 'circular', 'random'

"""

import numpy as np
from gymnasium.utils import EzPickle
from pettingzoo.utils.conversions import parallel_wrapper_fn

from mpe2._mpe_utils.core import Agent, Landmark, World
from mpe2._mpe_utils.scenario import BaseScenario
from mpe2._mpe_utils.simple_env import SimpleEnv, make_env
import pygame
from stochastic.processes.continuous import BrownianMotion


class raw_env(SimpleEnv, EzPickle):
    def __init__(
        self,
        num_agents=10,
        num_food_sources=1,
        obs_type = 'old',
        flow = 'none',
        max_cycles=25,
        continuous_actions=False,
        render_mode=None,
        dynamic_rescaling=False,
        cam_range=75.0,
        agent_radius=1.09,
        agent_velocity=3,
    ):
        EzPickle.__init__(
            self,
            num_agents=num_agents,
            num_food_sources=num_food_sources,
            obs_type=obs_type,
            flow=flow,
            max_cycles=max_cycles,
            continuous_actions=continuous_actions,
            render_mode=render_mode,
            cam_range=cam_range,
            agent_radius=agent_radius,
            agent_velocity=agent_velocity,
        )
        scenario = Scenario()
        world = scenario.make_world(num_agents, num_food_sources, obs_type, flow, cam_range, agent_radius, agent_velocity)
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
        self.cam_range = 75.0 # -75.0 in the negative direction and both, in x and y direction

        self.agent_trails = {f"agent_{i}": [] for i in range(num_agents)}  # Initialize trails for each agent
    
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

        # flip entities so landmarks are drawn last (they will be rendered in the background)
        entities_flipped = self.world.entities[::-1]

        # We want to prohibit agents moving outside the screen, so we choose a fixed cam_range
        cam_range = self.cam_range

        # Draw entities (landmarks and agents)
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

            # scale sizes as well
            radius = (entity.size / cam_range) * self.width // 2

            pygame.draw.circle(self.screen, entity.color * 200, (x, y), radius)
            pygame.draw.circle(self.screen, (0, 0, 0), (x, y), radius, 1)  # borders
            assert (
                0 < x < self.width and 0 < y < self.height
            ), f"Coordinates {(x, y)} are out of bounds."

        # Draw trails for each agent (rendered in front of entities)
        for agent_name, trail in self.agent_trails.items():
            if len(trail) > 1:
                scaled_trail = [
                    (
                        (pos[0] / cam_range) * self.width // 2 * 0.9 + self.width // 2,
                        (-pos[1] / cam_range) * self.height // 2 * 0.9 + self.height // 2,
                    )
                    for pos in trail
                ]
                # Find the agent by name
                agent = next(agent for agent in self.world.agents if agent.name == agent_name)
                pygame.draw.lines(self.screen, agent.color * 200, False, scaled_trail, 2)
    
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
                -self.cam_range,
                self.cam_range
            )

        # gather rewards
        for agent in self.world.agents:
            self.rewards[agent.name] = float(self.scenario.reward(agent, self.world))
        
        # Update trails for each agent
        for agent in self.world.agents:
            self.agent_trails[agent.name].append(agent.state.p_pos.copy())
            # Limit trail length to avoid memory issues
            if len(self.agent_trails[agent.name]) > 50:  # Keep the last 50 positions
                self.agent_trails[agent.name].pop(0)
    
    # set env action for a particular agent (Overwritten from mpe2/_mpe_utils/simple_env.py)
    def _set_action(self, action, agent, action_space, time=None):
        agent.action.u = np.zeros(self.world.dim_p)
        agent.action.c = np.zeros(self.world.dim_c)

        if agent.movable:
            # physical action
            agent.action.u = np.zeros(self.world.dim_p)
            if self.continuous_actions:
                # Process continuous action as in OpenAI MPE
                # Note: this ordering preserves the same movement direction as in the discrete case
                agent.action.u[0] += action[0][2] - action[0][1]
                agent.action.u[1] += action[0][4] - action[0][3]
            else:
                # process discrete action
                if action[0] == 1:
                    agent.action.u[0] = -1.0  # left
                elif action[0] == 2:
                    agent.action.u[0] = +1.0  # right
                elif action[0] == 3:
                    agent.action.u[1] = -1.0  # down
                elif action[0] == 4:
                    agent.action.u[1] = +1.0  # up
                elif action[0] == 0:
                    pass  # do nothing
            print('Action: ', action[0])
            action = action[1:]
        if not agent.silent:
            # communication action
            if self.continuous_actions:
                agent.action.c = action[0]
            else:
                agent.action.c = np.zeros(self.world.dim_c)
                agent.action.c[action[0]] = 1.0
            action = action[1:]
        # make sure we used all elements of action
        assert len(action) == 0


env = make_env(raw_env) # this throws an error for the return of the overwritten reset function (only for "env", not for "raw-env" or "parallel_env")
#env = raw_env
parallel_env = parallel_wrapper_fn(env)

class Scenario(BaseScenario):
    def make_world(self, num_agents=10, num_food_sources=1, obs_type = 'old', flow='none', cam_range=75.0, agent_radius=1.09, agent_velocity=3):
        world = MyWorld()
        num_agents = num_agents
        num_food_sources = num_food_sources
        world.obs_type = obs_type
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = f"agent_{i}"
            agent.collide = True # they will be propulsed away if they collide -> check, why this happens at the t+1 right now!
            agent.size = agent_radius
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
            landmark.size = 25 * agent_radius
            landmark.boundary = False # If True, landmark cannot be seen in observations
        world.cam_range = cam_range
        world.agent_velocity = agent_velocity
        # remove simulation timestep
        world.dt = 1.0
        # remove damping (loss of physical energy)
        self.damping = 0.0
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
            agent.state.p_pos = np_random.uniform(-world.cam_range, +world.cam_range, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p) # change p_vel if you want continuous movement
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np_random.uniform(-0.8*world.cam_range, +0.8*world.cam_range, world.dim_p)
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
        dist_to_food = np.sqrt(np.sum(np.square(agent.state.p_pos - world.landmarks[0].state.p_pos)))
        return -dist_to_food

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

    def observation(self, agent, world):
        # get distances to all other entities for this agent and combine with its velocity
        entity_pos = []
        
        if world.obs_type == "god":
            for entity in world.entities:
                if entity.boundary:  # invisible entities
                    entity_pos.append(np.array([0.0,0.0]))
                else:
                    entity_pos.append(entity.state.p_pos) # entity_pos are the vectors from the origin to the other entities
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
        # set applied forces (action plus Brownian noise)
        for i, agent in enumerate(self.agents):
            if agent.movable:
                noise = (self.apply_brownian_noise(num_dimensions=2)
                    if agent.u_noise
                    else 0.0)
                p_force[i] = agent.action.u * self.agent_velocity + 0.0 # instead, we can add the noise here
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

        # integrate physical state (this also applies environment (collision) forces now)
        self.integrate_state(p_force)
    
    # Override integrate_state function from _mpe2_utils/core.py
    # integrate physical state, including collision handling
    def integrate_state(self, p_force):
        # First, save all new positions and velocities in a temporary array
        new_pos = np.zeros((len(self.entities), self.dim_p))
        new_vels = np.zeros((len(self.entities), self.dim_p))
        for i, entity in enumerate(self.entities):
            if not entity.movable:
                new_vels[i] = entity.state.p_vel
                new_pos[i] = entity.state.p_pos
                continue
            new_vels[i] = entity.state.p_vel*(1 - self.damping)
            if p_force[i] is not None:
                new_vels[i] += p_force[i] * self.dt
            new_pos[i] = entity.state.p_pos + new_vels[i] * self.dt

            # Clip positions to stay within bounds of the window
            if new_pos[i][0] < -self.cam_range or new_pos[i][0] > self.cam_range or new_pos[i][1] < -self.cam_range or new_pos[i][1] > self.cam_range:
                new_pos[i] = np.clip(new_pos[i], -self.cam_range,self.cam_range) # change this to cam_range later
            # Add Brownian motion noise to not get stuck at borders
            #new_pos[i] += self.apply_brownian_noise(num_dimensions=2)

        # Next, change states until no collisions are detected anymore
        is_collision = True
        while is_collision:
            is_collision = False

            for a, pos_a in enumerate(new_pos):
                if self.entities[a].movable == False:
                    continue
                for b, pos_b in enumerate(new_pos):
                    if self.entities[b].movable == False or b<=a:
                        continue
                    # compute actual distance between entities
                    delta_pos = new_pos[a] - new_pos[b]
                    # compute Euclidean distance
                    dist = np.sqrt(np.sum(np.square(delta_pos)))
                    dist = max(dist, 0.1)
                    # minimum allowable distance
                    dist_min = self.entities[a].size + self.entities[b].size
                    # If collision detected, shift update position and velocity
                    if dist < dist_min and a != b:
                        missing_dist = dist_min - dist
                        # account for very small distances
                        delta_pos = np.array([max(delta_pos[0], 0.1), max(delta_pos[1], 0.1) ])
                        # unit direction from b → a
                        dir_ab = delta_pos  / dist
                        # displacement needed (but at least 25% of dist_min to avoid getting stuck with infinetely small corrections)
                        correction_vector = dir_ab * max(missing_dist, dist_min*0.25)
                        # update pos and vel
                        new_vels[a] += correction_vector
                        new_pos[a] += correction_vector
                        # Clip positions to stay within bounds of the window
                        if new_pos[a][0] < -self.cam_range or new_pos[a][0] > self.cam_range or new_pos[a][1] < -self.cam_range or new_pos[a][1] > self.cam_range:
                            new_pos[a] = np.clip(new_pos[a], -self.cam_range, self.cam_range)
                            # Add Brownian motion noise to not get stuck at borders
                            new_pos[a] += self.apply_brownian_noise(num_dimensions=2)
                        # set flag to true to indicate another check is needed
                        is_collision = True

        # Finally, save all new positions and velocities to the actual entities
        for i, entity in enumerate(self.entities):
            entity.state.p_vel = new_vels[i]
            entity.state.p_pos = new_pos[i]


    # Apply Brownian Noise (from a normal distribution with mean 0 and std 20% of velocity -> most values will be between -0.6*vel and 0.6*vel), multiplied by a small scalar
    def apply_brownian_noise(self, num_dimensions=2):
        brownian_step = []
        for dim in range(num_dimensions):
            brownian_step.append(BrownianMotion(drift=0, scale=0.2*self.agent_velocity, t=1, rng=None).sample_at([1])[0])
        return np.asarray(brownian_step)