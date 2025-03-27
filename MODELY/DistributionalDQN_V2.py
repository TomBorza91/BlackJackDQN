import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import time
import sys
import os
from collections import deque
import copy
import pickle  # ukládání a načítání souborů
import BlackJack
from matplotlib import pyplot as plt

class DistributionalDQN_V2(nn.Module):
    """
    Pokročilý Double Distributional  DQN agent pro hru BlackJack.

    Tento agent je založen na algoritmu Categorical DQN (C51), který modeluje distribuci návratnosti místo její střední hodnoty.
    Kombinuje přístup Double DQN pro redukci přeceňování Q-hodnot a distribuční přístup pro stabilnější a informativnější trénink.
    """

    def __init__(self, state_size, action_size, device, num_atoms=51, v_min=-10.0, v_max=10.0):
        """
        Inicializuje instanci pokročilého agenta Distributional Double DQN.

        Args:
            state_size (int): Počet vstupních vlastností (např. 24 pro pokročilý model).
            action_size (int): Počet dostupných akcí (např. 5 pro Blackjack).
            device (torch.device): Výpočetní zařízení ('cpu' nebo 'cuda').
            num_atoms (int, optional): Počet diskrétních atomů pro distribuci návratnosti. Defaultně 51.
            v_min (float, optional): Minimální hodnota distribuční podpory. Defaultně -10.0.
            v_max (float, optional): Maximální hodnota distribuční podpory. Defaultně 10.0.
        """
        super(DistributionalDQN_V2, self).__init__()
        self.device = device
        self.state_size = state_size
        self.action_size = action_size
        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.delta_z = (v_max - v_min) / (num_atoms - 1)
        self.support = torch.linspace(v_min, v_max, num_atoms).to(device)
        self.columns_to_normalize = list(range(24))
        self.GAMMA = 0.99
        self.LR =  0.001
        self.losses = []
        self.EPSILON = 1
        self.EPSILON_MIN = 0.01
        self.EPSILON_DECAY = 0.99999
        self.memory = deque(maxlen=250000)

        # PyTorch model definition
        self.q_network = self.build_model().to(device)
        self.q_network.apply(self.initialize_weights)  # Inicializace vah
        self.target_network = copy.deepcopy(self.q_network).to(device)
#        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.LR)
        self.optimizer = optim.SGD(self.q_network.parameters(), lr=self.LR, momentum=0.9)

        # Normalization bounds
        self.mn_total = 4
        self.mx_total = 30
        self.mn_d_show = 1
        self.mx_d_show = 30
        self.mn_cards_dealt = 4
        self.mx_cards_dealt = 416
        self.mn_hilo_idx = 100
        self.mx_hilo_idx = -100
        self.mn_card_seen = 0
        self.mx_card_seen = 32
        self.mn_card_10_seen = 0
        self.mx_card_10_seen = 128
        self.mx_lo = 160
        self.mn_lo = 0
        self.mx_mid = 96
        self.mn_mid = 0
        self.mx_hi = 160
        self.mn_hi = 0
        self.mx_mnmx_ratio = 160
        self.mn_mnmx_ratio = 0
        self.mx_lo_ratio = 160
        self.mn_lo_ratio = 0
        self.mx_mid_ratio = 96
        self.mn_mid_ratio = 0
        self.mx_hi_ratio = 160
        self.mn_hi_ratio = 0
        self.mn_cards_left = 0
        self.mx_cards_left = 416
        self.mn_21diff = 0
        self.mx_21diff = 21
        self.mn_split_total = 0
        self.mx_split_total = 11
        self.min_vals = np.array([self.mn_total, self.mn_d_show, self.mn_cards_dealt, self.mn_hilo_idx, self.mn_card_seen, self.mn_card_seen, self.mn_card_seen, self.mn_card_seen, self.mn_card_seen, self.mn_card_seen, self.mn_card_seen, self.mn_card_seen, self.mn_card_10_seen, self.mn_card_seen, self.mn_lo, self.mn_mid, self.mn_hi, self.mn_mnmx_ratio, self.mn_lo_ratio, self.mn_mid_ratio, self.mn_hi_ratio, self.mn_cards_left, self.mn_21diff, self.mn_split_total], dtype=float)
        self.max_vals = np.array([self.mx_total, self.mx_d_show, self.mx_cards_dealt, self.mx_hilo_idx, self.mx_card_seen, self.mx_card_seen, self.mx_card_seen, self.mx_card_seen, self.mx_card_seen, self.mx_card_seen, self.mx_card_seen, self.mx_card_seen, self.mx_card_10_seen, self.mx_card_seen, self.mx_lo, self.mx_mid, self.mx_hi, self.mx_mnmx_ratio, self.mx_lo_ratio, self.mx_mid_ratio, self.mx_hi_ratio, self.mx_cards_left, self.mx_21diff, self.mx_split_total], dtype=float)
        self.min_reward = np.array([-2.5], dtype=float)
        self.max_reward = np.array([2], dtype=float)


    def build_model(self):
        """
        Vytvoří neuronovou síť pro predikci distribuce přes atomy.

        Returns:
            nn.Sequential: Architektura sítě.
        """
        return nn.Sequential(
            nn.Linear(self.state_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.action_size * self.num_atoms)
        )

    def initialize_weights(self, m):
        """
        Inicializuje váhy neuronové sítě pomocí Kaiming uniform inicializace.

        Args:
            m (nn.Module): Vrstva sítě.
        """
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x, network_type='q_network'):
        """
        Výpočet výstupní distribuce a očekávaných Q-hodnot.

        Args:
            x (Tensor): Vstupní stav(y).
            network_type (str): Typ použité sítě ('q_network' nebo 'target_network').

        Returns:
            tuple: (q_values [Tensor], q_probs [Tensor])
                - q_values: Očekávané hodnoty akcí.
                - q_probs: Distribuce pravděpodobností pro každou akci.
        """
        if network_type == 'q_network':
            network = self.q_network
        elif network_type == 'target_network':
            network = self.target_network
        else:
            raise ValueError("Invalid network type. Choose 'q_network' or 'target_network'.")

        q_atoms = network(x).view(-1, self.action_size, self.num_atoms)
        q_probs = torch.softmax(q_atoms, dim=2)  # Softmax over atoms
        q_values = torch.sum(q_probs * self.support, dim=2)  # Expected value
        return q_values, q_probs

    def normalize_states(self, states):
        """
        Normalizuje vstupní stavy do rozsahu [0, 1].

        Args:
            states (np.ndarray): Numpy pole stavů.

        Returns:
            np.ndarray: Normalizované stavy.
        """
        norm_states = states.copy().astype(float)
        norm_states[:, self.columns_to_normalize] = (states[:, self.columns_to_normalize] - self.min_vals) / (self.max_vals - self.min_vals)
        return norm_states

    def player_action(self, state, action_mask=None):
        """
        Vybere nejlepší možnou akci na základě predikovaných Q hodnot.

        Args:
            state (np.ndarray): Stav ve formátu (1, state_size).
            action_mask (np.ndarray): Maska platných akcí (0 nebo 1).

        Returns:
            int: Index vybrané akce.
        """
        norm_state = self.normalize_states(state)
        norm_state = torch.FloatTensor(norm_state).to(self.device)
        with torch.no_grad():
            q_values, _ = self.forward(norm_state)
        if action_mask is not None:
            action_mask_tensor = torch.FloatTensor(action_mask).to(self.device)
            q_values[0] -= (action_mask_tensor - 1) * -1e12
        return torch.argmax(q_values).item()

    def action(self, state, action_mask=None, train=True):
        """
        Vybere akci pomocí epsilon-greedy strategie.

        Args:
            state (np.ndarray): Aktuální stav.
            action_mask (np.ndarray): Maska možných akcí.
            train (bool): Indikuje, zda se jedná o trénovací režim.

        Returns:
            int: Index zvolené akce.
        """
        norm_state = self.normalize_states(state)
        norm_state = torch.FloatTensor(norm_state).to(self.device)
        with torch.no_grad():
            q_values, _ = self.forward(norm_state)
        if train and random.random() < self.EPSILON:
            possible_actions = [action for action in range(self.action_size) if action_mask[action] == 1]
            return random.choice(possible_actions)
        else:
            if action_mask is not None:
                action_mask_tensor = torch.FloatTensor(action_mask).to(self.device)
                q_values[0] -= (action_mask_tensor - 1) * -1e12
            return torch.argmax(q_values).item()

    def train(self, batch_size):
        """
        Provede jedno trénovací kolo nad minibatchí zkušeností z paměti.

        Args:
            batch_size (int): Velikost minibatche.
        """
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)
        states = np.array([x[0][0] for x in minibatch])
        next_states = np.array([x[3][0] for x in minibatch])
        actions = np.array([x[1] for x in minibatch])
        rewards = np.array([x[2] for x in minibatch])
        dones = np.array([x[4] for x in minibatch], dtype=float)

        # Normalizace odměn
        rewards_norm = ((rewards - self.min_reward[0]) / (self.max_reward[0] - self.min_reward[0])) * 10

        # Normalizace stavů
        states_norm = self.normalize_states(states)
        next_states_norm = self.normalize_states(next_states)

        # Konverze na tensory
        states_tensor = torch.FloatTensor(states_norm).to(self.device)
        next_states_tensor = torch.FloatTensor(next_states_norm).to(self.device)
        actions_tensor = torch.LongTensor(actions).to(self.device)
        rewards_tensor = torch.FloatTensor(rewards_norm).to(self.device)
        dones_tensor = torch.FloatTensor(dones).to(self.device)

        # Target distribuce
        with torch.no_grad():
            next_q_values, next_probs = self.forward(next_states_tensor,'target_network')
            _, next_q_probs = self.forward(next_states_tensor,'q_network')
            best_actions = torch.argmax(torch.sum(next_q_probs * self.support, dim=2), dim=1)
            target_probs = next_probs[range(batch_size), best_actions]

        projected_distribution = self.project_distribution(target_probs, rewards_tensor, dones_tensor)

        # Aktuální distribuce
        _, predicted_probs = self.forward(states_tensor,'q_network')
        predicted = predicted_probs[range(batch_size), actions_tensor]
        loss = F.kl_div(
            torch.log(predicted + 1e-8), projected_distribution, reduction='batchmean'
        )
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Logování ztrát a aktualizace target network
        self.losses.append(loss.item())
        self.update_target_network()

    def project_distribution(self, target_probs, rewards, dones):
        """
        Převádí distribuční cílovou hodnotu na podporu atomů.

        Args:
            target_probs (Tensor): Pravděpodobnosti z cílové sítě.
            rewards (Tensor): Odměny.
            dones (Tensor): Příznak terminálního stavu.

        Returns:
            Tensor: Projektovaná distribuce pro každý vzorek v batchi.
        """
        batch_size = rewards.shape[0]

        # Výpočet distribuce na základě odměn a support atomů
        tz = rewards[:, None] + (1 - dones[:, None]) * self.GAMMA * self.support

        # Omezení rozsahu distribuce na [v_min, v_max]
        tz = tz.clamp(self.v_min, self.v_max)

        # Výpočet indexů atomů
        b = (tz - self.v_min) / self.delta_z
        l = b.floor().long()
        u = b.ceil().long()
        # Inicializace prázdné distribuce
        batch_indices = torch.arange(batch_size, device=self.device).unsqueeze(1)
        proj_dist = torch.zeros((batch_size, self.num_atoms), device=self.device)

        # Speciální případ: přesný zásah
    # Speciální případ: přesný zásah
        exact_match_mask = (l == u)
        if exact_match_mask.any():
          exact_indices = exact_match_mask.nonzero(as_tuple=True)
          proj_dist[exact_indices[0], l[exact_match_mask]] += target_probs[exact_match_mask]

        # Obecný případ: rozdělení mezi sousedy
        not_exact_mask = ~exact_match_mask
        if not_exact_mask.any():
          not_exact_indices = not_exact_mask.nonzero(as_tuple=True)
          proj_dist[not_exact_indices[0], l[not_exact_mask]] += target_probs[not_exact_mask] * (u[not_exact_mask] - b[not_exact_mask])
          proj_dist[not_exact_indices[0], u[not_exact_mask]] += target_probs[not_exact_mask] * (b[not_exact_mask] - l[not_exact_mask])

        proj_dist /= proj_dist.sum(dim=1, keepdim=True).clamp(min=1e-8)
        return proj_dist

    def save_memory_file(self):
        """
        Uloží replay paměť na disk.
        """
        with open('/content/drive/MyDrive/Diplomka/model/DistDQN_V2/memory.pkl', 'wb') as f:
            pickle.dump(self.memory, f)

    def save_memory(self, state, action, reward, next_state, done):
        """
        Uloží jednu zkušenost do paměti.

        Args:
            state (np.ndarray): Původní stav.
            action (int): Vybraná akce.
            reward (float): Odměna za akci.
            next_state (np.ndarray): Nový stav.
            done (bool): Indikátor konce epizody.
        """
        self.memory.append((state, action, reward, next_state, done))

    def load_model(self):
        """
        Načte model, epsilon a paměť z disku.
        """
        self.q_network.load_state_dict(torch.load('/content/drive/MyDrive/Diplomka/model/DistDQN_V2/q_network_complete.pt'))
        self.target_network.load_state_dict(torch.load('/content/drive/MyDrive/Diplomka/model/DistDQN_V2/q_network_complete.pt'))
        with open('/content/drive/MyDrive/Diplomka/model/DistDQN_V2/model_params.pkl', 'rb') as f:
            params = pickle.load(f)
            self.EPSILON = params['epsilon']
        with open('/content/drive/MyDrive/Diplomka/model/DistDQN_V2/memory.pkl', 'rb') as f:
            self.memory = deque(pickle.load(f), maxlen=250000)
        print("Model and parameters loaded successfully.")

    def update_target_network(self, tau=0.005):
        """
        Provádí soft update cílové sítě.

        Args:
            tau (float): Koeficient interpolace mezi váhami sítí.
        """
        for target_param, param in zip(self.target_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

    def save_model(self):
        """
        Uloží model, epsilon a paměť na disk.
        """
        torch.save(self.q_network.state_dict(), '/content/drive/MyDrive/Diplomka/model/DistDQN_V2/q_network_complete.pt')
        with open('/content/drive/MyDrive/Diplomka/model/DistDQN_V2/model_params.pkl', 'wb') as f:
            pickle.dump({'epsilon': self.EPSILON}, f)
        with open('/content/drive/MyDrive/Diplomka/model/DistDQN_V2/memory.pkl', 'wb') as f:
            pickle.dump(self.memory, f)

    def update_learning_rate(self, new_learning_rate):
        """
        Aktualizuje učící rychlost optimizeru.

        Args:
            new_learning_rate (float): Nová hodnota learning rate.
        """
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_learning_rate

    def update_epsilon(self, new_epsilon):
        """
        Aktualizuje epsilon hodnotu pro průzkum.

        Args:
            new_epsilon (float): Nová hodnota epsilonu.
        """
        self.EPSILON = new_epsilon

    def epsilon_decay(self):
        """
        Postupně snižuje epsilon podle decay faktoru.
        """
        if self.EPSILON > self.EPSILON_MIN:
            self.EPSILON *= self.EPSILON_DECAY

    def save_model_checkpoint(self, round_count):
        """
        Uloží aktuální stav modelu jako checkpoint.

        Args:
            round_count (int): Počet odehraných kol (pro pojmenování).
        """
        checkpoint_dir = f'/content/drive/MyDrive/Diplomka/model/DistDQN_V2/checkpoint/{round_count}/'
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Uložení modelu
        torch.save(self.q_network.state_dict(), f'{checkpoint_dir}q_network_checkpoint.pt')

        # Uložení dalších parametrů a paměti
        with open(f'{checkpoint_dir}model_params.pkl', 'wb') as f:
            pickle.dump({
                'epsilon': self.EPSILON,
                'memory': self.memory
            }, f)
        with open(f'{checkpoint_dir}memory.pkl', 'wb') as f:
            pickle.dump(self.memory, f)