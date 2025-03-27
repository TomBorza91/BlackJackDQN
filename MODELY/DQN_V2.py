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

class DQN_V2(nn.Module):
    """
    Pokročilý agent pro hru BlackJack využívající algoritmus Double DQN.

    Tento model používá hlubší neuronovou síť a rozšířený stavový vektor (27 vstupních hodnot),
    zahrnující Hi-Lo index, rozdělení karet na low/mid/high, počet rozdáných karet, vzdálenost od 21 atd.

    Využívá target network, experience replay, epsilon-greedy výběr akcí a soft update cílové sítě.
    """
    def __init__(self, state_size, action_size, device):
        """
        Inicializace agenta.

        Args:
            state_size (int): Počet vstupních hodnot (features).
            action_size (int): Počet možných akcí.
            device (str): Výpočetní zařízení ('cpu' nebo 'cuda').
        """
        super(DQN_V2, self).__init__()
        self.device = device
        self.state_size = state_size
        self.action_size = action_size
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
        self.q_network.apply(self.initialize_weights)  # Aplikace inicializace vah
        self.target_network = copy.deepcopy(self.q_network).to(device)
       # self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.LR)
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

    def initialize_weights(self, m):
        """
        Inicializuje váhy lineárních vrstev pomocí Kaiming uniform inicializace.

        Args:
            m (nn.Module): Vrstva, která se inicializuje.
        """        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def build_model(self):
        """
        Vytváří architekturu neuronové sítě.

        Returns:
            nn.Sequential: Definovaná síť se třemi skrytými vrstvami a ReLU aktivací.
        """
        return nn.Sequential(
            nn.Linear(self.state_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.action_size)
        )

    def forward(self, x):
        """
        Provede forward průchod přes neuronovou síť.

        Args:
            x (torch.Tensor): Vstupní tensor.

        Returns:
            torch.Tensor: Q-hodnoty pro všechny možné akce.
        """
        return self.q_network(x)

    def normalize_states(self, states):
        """
        Normalizuje vstupní stavy dle předdefinovaných rozsahů.

        Args:
            states (np.ndarray): Stavy tvaru (batch_size, state_size).

        Returns:
            np.ndarray: Normalizované stavy.
        """
        norm_states = states.copy().astype(float)
        norm_states[:, self.columns_to_normalize] = (states[:, self.columns_to_normalize] - self.min_vals) / (self.max_vals - self.min_vals)
        return norm_states

    def player_action(self, state, action_mask=None):
        """
        Vybere nejlepší akci (bez průzkumu), vhodné pro inference.

        Args:
            state (np.ndarray): Vstupní stav (1 x state_size).
            action_mask (np.ndarray, optional): Binární maska platných akcí.

        Returns:
            int: Index akce s nejvyšší Q-hodnotou.
        """
        norm_state = self.normalize_states(state)
        norm_state = torch.FloatTensor(norm_state).to(self.device)
        with torch.no_grad():
            action_values = self.q_network(norm_state)
        if action_mask is not None:
            action_mask_tensor = torch.FloatTensor(action_mask).to(self.device)
            action_values[0] -= (action_mask_tensor - 1) * -1e12
        return torch.argmax(action_values).item()

    def action(self, state, action_mask=None, train=True):
        """
        Vybere akci podle epsilon-greedy strategie.

        Args:
            state (np.ndarray): Aktuální herní stav.
            action_mask (np.ndarray, optional): Maska povolených akcí.
            train (bool): Určuje, zda aplikovat epsilon průzkum (True při trénování).

        Returns:
            int: Index vybrané akce.
        """
        norm_state = self.normalize_states(state)
        norm_state = torch.FloatTensor(norm_state).to(self.device)
        if action_mask  is not None:
          action_mask_tensor = torch.FloatTensor(action_mask).to(self.device)
        if train and random.random() < self.EPSILON:
            possible_actions = [action for action in range(self.action_size) if action_mask[action] == 1]
            return random.choice(possible_actions)
        else:
            with torch.no_grad():
                action_values = self.q_network(norm_state)
            if action_mask is not None:
                action_values[0] -= (action_mask_tensor - 1) * -1e12
            return torch.argmax(action_values).item()

    def save_memory_file(self):
        """
        Uloží paměť (replay buffer) do souboru.
        """
        with open('/content/drive/MyDrive/Diplomka/model/DoubleDQN_V2/memory.pkl', 'wb') as f:
            pickle.dump(self.memory, f)

    def save_memory(self, state, action, reward, next_state, done):
        """
        Uloží zkušenost do paměti.

        Args:
            state (np.ndarray): Aktuální stav.
            action (int): Provedená akce.
            reward (float): Získaná odměna.
            next_state (np.ndarray): Stav po akci.
            done (bool): True pokud epizoda skončila.
        """
        self.memory.append((state, action, reward, next_state, done))

    def load_model(self):
        """
        Načte model, paměť a parametry z disku.
        """
        self.q_network.load_state_dict(torch.load('/content/drive/MyDrive/Diplomka/model/DoubleDQN_V2/q_network_complete.pt'))
        self.target_network.load_state_dict(torch.load('/content/drive/MyDrive/Diplomka/model/DoubleDQN_V2/q_network_complete.pt'))
        with open('/content/drive/MyDrive/Diplomka/model/DoubleDQN_V2/model_params.pkl', 'rb') as f:
            params = pickle.load(f)
            self.EPSILON = params['epsilon']
        with open('/content/drive/MyDrive/Diplomka/model/DoubleDQN_V2/memory.pkl', 'rb') as f:
            self.memory = deque(pickle.load(f), maxlen=250000)
        print("Model and parameters loaded successfully.")

    def train(self, batch_size):
        """
        Provádí učení z náhodného vzorku z paměti.

        Args:
            batch_size (int): Počet vzorků v minibatchi.
        """
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        states = np.array([x[0][0] for x in minibatch])
        next_states = np.array([x[3][0] for x in minibatch])

        states_norm = self.normalize_states(states)
        next_states_norm = self.normalize_states(next_states)

        states_tensor = torch.FloatTensor(states_norm).to(self.device)
        next_states_tensor = torch.FloatTensor(next_states_norm).to(self.device)

        predicted = self.q_network(states_tensor)
        next_predicted = self.target_network(next_states_tensor)

        targets = predicted.clone().detach()
        actions = np.array([x[1] for x in minibatch])
        rewards = np.array([x[2] for x in minibatch])
        rewards_norm = rewards.astype(float) / abs(self.min_reward)
        dones = np.array([x[4] for x in minibatch])

        targets[range(batch_size), actions] = torch.FloatTensor(rewards_norm).to(self.device) + self.GAMMA * torch.max(next_predicted, dim=1)[0] * (1 - torch.FloatTensor(dones).to(self.device))

        self.optimizer.zero_grad()
        loss = nn.MSELoss()(predicted, targets)
        loss.backward()
        self.optimizer.step()

        self.losses.append(loss.item())
        self.update_target_network()

    def update_target_network(self, tau=0.001):
        """
        Provede soft update cílové sítě směrem k hlavní síti.

        Args:
            tau (float): Míra přiblížení (default 0.001).
        """
        for target_param, param in zip(self.target_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

    def save_model(self):
        """
        Uloží trénovaný model a parametry (např. epsilon) na disk.
        """
        torch.save(self.q_network.state_dict(), '/content/drive/MyDrive/Diplomka/model/DoubleDQN_V2/q_network_complete.pt')
        with open('/content/drive/MyDrive/Diplomka/model/DoubleDQN_V2/model_params.pkl', 'wb') as f:
            pickle.dump({'epsilon': self.EPSILON}, f)
        with open('/content/drive/MyDrive/Diplomka/model/DoubleDQN_V2/memory.pkl', 'wb') as f:
            pickle.dump(self.memory, f)

    def update_learning_rate(self, new_learning_rate):
        """
        Nastaví novou hodnotu learning rate.

        Args:
            new_learning_rate (float): Nová hodnota learning rate.
        """
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_learning_rate

    def update_epsilon(self, new_epsilon):
        """
        Ručně nastaví novou hodnotu epsilon.

        Args:
            new_epsilon (float): Nová hodnota epsilon.
        """
        self.EPSILON = new_epsilon

    def epsilon_decay(self):
        """
        Aplikuje exponenciální snížení hodnoty epsilon, pokud ještě neklesla na minimum.
        """
        if self.EPSILON > self.EPSILON_MIN:
            self.EPSILON *= self.EPSILON_DECAY

    def save_model_checkpoint(self, round_count):
        """
        Uloží checkpoint modelu v daném kole.

        Args:
            round_count (int): Počet odehraných kol (slouží k pojmenování adresáře).
        """        checkpoint_dir = f'/content/drive/MyDrive/Diplomka/model/DoubleDQN_V2/checkpoint/{round_count}/'
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