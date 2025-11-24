###############################################################
# 0. Imports
###############################################################

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical, Normal
import numpy as np
import gymnasium as gym
import kymnasium as kym
from typing import Any, Dict, List
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


###############################################################
# 1. Normalization Utilities
###############################################################

def normalize_position(x, y, R):
    return x / R, y / R

def normalize_velocity(vx, vy):
    return np.clip(vx, -1, 1), np.clip(vy, -1, 1)

def normalize_angle(angle_deg):
    rad = angle_deg * np.pi / 180
    return np.sin(rad), np.cos(rad)

def normalize_power(power):
    return power / 2500


###############################################################
# 2. Observation Parsing
###############################################################

def parse_obs(obs, prev_obs=None, board_R=1.0):
    """
    obs: dict from kymnasium AlKkaGi env
    returns:
      stone_states: list of 3 stone-specific observations
      full_state: global observation
    """

    turn = obs["turn"]
    my = obs["black"] if turn == 0 else obs["white"]
    opp = obs["white"] if turn == 0 else obs["black"]
    obstacles = obs["obstacles"]

    stone_states = []

    for i in range(3):
        x, y, alive = my[i]
        x_n, y_n = normalize_position(x, y, board_R)

        if prev_obs is not None:
            px, py, _ = (prev_obs["black"] if turn==0 else prev_obs["white"])[i]
            vx, vy = normalize_velocity(x - px, y - py)
        else:
            vx, vy = 0.0, 0.0

        opp_flat = np.array(opp).flatten()
        obs_flat = np.array(obstacles).flatten()

        stone_state = np.concatenate([
            [x_n, y_n, alive, vx, vy],
            opp_flat,
            obs_flat,
            [turn]
        ])
        stone_states.append(torch.tensor(stone_state, dtype=torch.float32))

    # full observation
    full_flat = np.concatenate([
        np.array(my).flatten(),
        np.array(opp).flatten(),
        np.array(obstacles).flatten(),
        [turn]
    ])

    full_state = torch.tensor(full_flat, dtype=torch.float32)

    return stone_states, full_state


###############################################################
# 3. Risk Calculator (GT for Stage1)
###############################################################

def compute_risk(stone_state):
    """
    stone_state: torch tensor of processed features
    RETURN: scalar risk value (float)
    """
    # TODO: implement full risk calculation from previous design
    # Placeholder:
    risk = torch.rand(1)
    return risk


###############################################################
# 4. Proposal Quality Calculator (GT for Stage2)
###############################################################

def compute_proposal_quality(before_state, after_state, proposed_action):
    """
    before_state, after_state: full board states for GT calculation
    proposed_action: (power, angle) raw proposals
    RETURN: scalar proposal quality
    """
    # TODO: implement full GT proposal quality logic
    # Placeholder:
    return torch.rand(1)


###############################################################
# 5. StoneAgent Network
###############################################################

class StoneAgent(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.stage1_head = nn.Linear(hidden_dim, 1)
        self.stage2_action = nn.Linear(hidden_dim, 2)    # raw power, raw angle
        self.stage2_priority = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        h = self.shared(x)
        score = self.stage1_head(h)
        action_raw = self.stage2_action(h)
        priority = self.stage2_priority(h)
        return score, action_raw, priority


###############################################################
# 6. MasterAgent Network
###############################################################

class MasterAgent(nn.Module):
    def __init__(self, input_dim, hidden_dim=256):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.index_head = nn.Linear(hidden_dim, 3)
        self.mu = nn.Linear(hidden_dim, 2)
        self.log_sigma = nn.Parameter(torch.zeros(2))

    def forward(self, x):
        h = self.shared(x)
        index_logits = self.index_head(h)
        mu = self.mu(h)
        sigma = torch.exp(self.log_sigma)
        return index_logits, mu, sigma


###############################################################
# 7. Critic Network
###############################################################

class Critic(nn.Module):
    def __init__(self, input_dim, hidden_dim=256):
        super().__init__()
        self.v = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.v(x)


###############################################################
# 8. Hierarchical Action Selection
###############################################################

def hierarchical_act(stone_agents, master_agent, stone_states, full_state):
    # Stage1: scores
    stage1_scores = []
    for i in range(3):
        score, _, _ = stone_agents[i](stone_states[i])
        stage1_scores.append(score)

    # Pick candidate via argmax of Stage1 scores
    candidate_idx = torch.argmax(torch.stack(stage1_scores)).item()

    # Stage2 proposals: all stone agents propose actions for the chosen stone
    stage2_outputs = []
    selected_state = stone_states[candidate_idx]
    for i in range(3):
        _, action_raw, priority = stone_agents[i](selected_state)
        stage2_outputs.append((action_raw, priority))

    # Prepare master input
    stage1_tensor = torch.stack(stage1_scores).squeeze()
    stage2_flat = torch.cat([
        torch.cat([out[0].squeeze(), out[1].squeeze()])
        for out in stage2_outputs
    ])

    master_input = torch.cat([full_state, stage1_tensor, stage2_flat])
    index_logits, mu, sigma = master_agent(master_input)

    # Sample from master
    index_dist = Categorical(logits=index_logits)
    final_index = index_dist.sample()

    cont_dist = Normal(mu, sigma)
    cont_action = cont_dist.sample()

    # Convert raw actions into actual angle/power
    final_power = torch.sigmoid(cont_action[0]) * 2500
    final_angle = torch.tanh(cont_action[1]) * 180

    return {
        "turn": 0,
        "index": final_index.item(),
        "power": final_power.item(),
        "angle": final_angle.item()
    }, (index_dist, cont_dist), stage1_scores, stage2_outputs


###############################################################
# 9. PPO Buffer
###############################################################

class PPOBuffer:
    def __init__(self):
        self.obs: List[torch.Tensor] = []
        self.actions: List[Dict[str, float]] = []
        self.logps: List[Any] = []
        self.rewards: List[float] = []
        self.values: List[float] = []
        self.dones: List[bool] = []

    def store(self, *args):
        self.obs.append(args[0])
        self.actions.append(args[1])
        self.logps.append(args[2])
        self.rewards.append(args[3])
        self.values.append(args[4])
        self.dones.append(args[5])

    def clear(self):
        self.__init__()


###############################################################
# 10. PPO Update (Skeleton)
###############################################################

def ppo_update(master_agent, stone_agents, critic, buffer, optimizer):
    # TODO: implement full PPO update logic
    # This skeleton defines the loop, but not full math.
    if not buffer.rewards:
        return

    _ = compute_advantages_stub(buffer)
    # TODO: compute losses for:
    # - master policy
    # - critic
    # - stone stage1 (risk prediction)
    # - stone stage2 (proposal quality)
    # For now, we simply clear gradients to keep the training loop functional.
    optimizer.zero_grad()
    optimizer.step()


###############################################################
# 11. Advantage Stub (to be implemented)
###############################################################

def compute_advantages_stub(buffer):
    # TODO: replace with GAE
    rewards = buffer.rewards
    dones = buffer.dones
    returns = []
    gamma = 0.99
    ret = 0.0
    for r, d in zip(reversed(rewards), reversed(dones)):
        if d:
            ret = 0.0
        ret = r + gamma * ret
        returns.insert(0, ret)
    return torch.tensor(returns, dtype=torch.float32)


###############################################################
# 12. Agent Setup
###############################################################

class _BaseAgent(kym.Agent):
    def __init__(self, turn: int):
        self.turn = turn

    def _choose_index(self, observation: Dict) -> int:
        stones = observation["black"] if self.turn == 0 else observation["white"]
        for idx, (_, _, alive) in enumerate(stones):
            if alive:
                return idx
        return 0

    def _default_action(self, observation: Dict) -> Dict[str, float]:
        return {
            "turn": self.turn,
            "index": self._choose_index(observation),
            "power": 1.0,
            "angle": 0.0,
        }

    @classmethod
    def load(cls, path: str) -> "kym.Agent":
        # Placeholder load for competition API compatibility
        turn = torch.load(path)[0] if path else 0
        return cls(turn)

    def save(self, path: str):
        torch.save((self.turn,), path)


class myBlackAgent(_BaseAgent):
    def __init__(self, turn: int = 0):
        super().__init__(turn=turn)

    def act(self, observation: Any, info: Dict):
        return self._default_action(observation)


class myWhiteAgent(_BaseAgent):
    def __init__(self, turn: int = 1):
        super().__init__(turn=turn)

    def act(self, observation: Any, info: Dict):
        return self._default_action(observation)


###############################################################
# 13. Training Loop (Skeleton)
###############################################################

def train():
    env = gym.make("kymnasium/AlKkaGi-3x3-v0", render_mode="rgb_array")

    stone_agents = [
        StoneAgent(input_dim=40),
        StoneAgent(input_dim=40),
        StoneAgent(input_dim=40)
    ]

    master_agent = MasterAgent(input_dim=80)
    critic = Critic(input_dim=80)

    optimizer = optim.Adam(
        list(master_agent.parameters())
        + list(critic.parameters())
        + sum([list(agent.parameters()) for agent in stone_agents], []),
        lr=3e-4
    )

    buffer = PPOBuffer()

    for episode in range(100000):
        obs, info = env.reset()
        prev_obs = None
        done = False

        while not done:
            stone_states, full_state = parse_obs(obs, prev_obs)
            action, dists, s1_scores, s2_outputs = hierarchical_act(
                stone_agents, 
                master_agent,
                stone_states,
                full_state
            )

            next_obs, env_r, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Compute GT for Stage1
            gt_risks = [compute_risk(s) for s in stone_states]

            # Compute GT for Stage2
            proposal_q = compute_proposal_quality(
                obs, next_obs, s2_outputs
            )

            final_reward = env_r

            buffer.store(full_state, action, dists, final_reward, 0, done)

            prev_obs = obs
            obs = next_obs

        ppo_update(master_agent, stone_agents, critic, buffer, optimizer)
        buffer.clear()
