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

class Dueling_DQN_V2(nn.Module):
    """
    Agent pro hru Blackjack využívající pokročilou architekturu Double Dueling DQN.

    Tento agent odhaduje hodnoty Q pomocí odděleného výpočtu hodnoty stavu (value stream)
    a výhodnosti jednotlivých akcí (advantage stream), čímž zlepšuje stabilitu a přesnost
    učení. Využívá také Double DQN strategii, kde se akce vybírá z online sítě a hodnota
    se bere z cílové (target) sítě.

    """
    def __init__(self, state_size, action_size, device):
        """
        Inicializuje instanci pokročilého agenta Double Dueling DQN pro Blackjack.

        Inicializuje:
            - online i cílovou síť (dueling architektura),
            - optimalizátor,
            - zkušenostní paměť (replay buffer),
            - parametry epsilon-greedy strategie,
            - hranice pro normalizaci vstupů a odměn.

        Args:
            state_size (int): Počet vstupních proměnných reprezentujících stav (např. skóre hráče, Hi-Lo index, počet karet atd.).
            action_size (int): Počet možných akcí (např. HIT, STAND, DOUBLE, SPLIT, INSURANCE).
            device (torch.device): Cílové zařízení pro výpočty ('cpu' nebo 'cuda').
        """
        super(Dueling_DQN_V2, self).__init__()
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

        # PyTorch model definition for Dueling DQN
        self.q_network = self.build_model().to(device)
        # Apply He Initialization
        self.apply_initialization()
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


    def build_model(self):
        """
        Vytvoří double dueling dqn architekturu sestávající z:
        - feature layer: sdílená část sítě
        - value stream: předpovídá hodnotu stavu V(s)
        - advantage stream: předpovídá výhodu jednotlivých akcí A(s, a)

        Returns:
            nn.ModuleDict: Složky neuronové sítě (feature, value, advantage).
        """
        feature_layer = nn.Sequential(
            nn.Linear(self.state_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )

        value_layer = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        advantage_layer = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.action_size)
        )

        return nn.ModuleDict({
            'feature_layer': feature_layer,
            'value_layer': value_layer,
            'advantage_layer': advantage_layer
        })

    def initialize_weights(self, m):
        """
        Inicializuje váhy vrstev pomocí He (Kaiming) inicializace.

        Args:
            m (nn.Module): Vrstva neuronové sítě.
        """
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def apply_initialization(self):
        """
        Aplikuje inicializaci na všechny vrstvy neuronové sítě.
        """
        for layer in self.q_network.values():
            layer.apply(self.initialize_weights)

    def forward(self, x, network_type='q_network'):
        """
        Výpočet Q hodnot pomocí dueling architektury.

        Args:
            x (torch.Tensor): Vstupní stav.
            network_type (str): Typ sítě – 'q_network' nebo 'target_network'.

        Returns:
            torch.Tensor: Odhadnuté Q hodnoty pro všechny akce.
        """
        if network_type == 'q_network':
            network = self.q_network
        elif network_type == 'target_network':
            network = self.target_network
        else:
            raise ValueError("Invalid network type. Choose 'q_network' or 'target_network'.")

        features = network['feature_layer'](x)
        value = network['value_layer'](features)
        advantage = network['advantage_layer'](features)
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values

    def normalize_states(self, states):
        """
        Normalizuje vstupní stavy do rozsahu 0–1 na základě předdefinovaných minim a maxim.

        Args:
            states (np.ndarray): Původní stavy (např. z prostředí).

        Returns:
            np.ndarray: Normalizované stavy.
        """
        norm_states = states.copy().astype(float)
        norm_states[:, self.columns_to_normalize] = (states[:, self.columns_to_normalize] - self.min_vals) / (self.max_vals - self.min_vals)
        return norm_states

    def player_action(self, state, action_mask=None):
        """
        Vrací nejlepší akci (greedy), vhodné např. pro testování modelu.

        Args:
            state (np.ndarray): Vstupní stav.
            action_mask (np.ndarray, optional): Maska platných akcí (1 = povoleno, 0 = zakázáno).

        Returns:
            int: Index vybrané akce.
        """
        norm_state = self.normalize_states(state)
        norm_state = torch.FloatTensor(norm_state).to(self.device)
        with torch.no_grad():
            action_values = self.forward(norm_state)
        if action_mask is not None:
            action_mask_tensor = torch.FloatTensor(action_mask).to(self.device)
            action_values[0] -= (action_mask_tensor - 1) * -1e12
        return torch.argmax(action_values).item()

    def action(self, state, action_mask=None, train=True):
        """
        Epsilon-greedy výběr akce – používá se během tréninku.

        Args:
            state (np.ndarray): Vstupní stav.
            action_mask (np.ndarray, optional): Maska platných akcí.
            train (bool): Indikuje, zda jsme ve fázi tréninku.

        Returns:
            int: Vybraná akce (index).
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
                action_values = self.forward(norm_state)
            if action_mask is not None:
                action_values[0] -= (action_mask_tensor - 1) * -1e12
            return torch.argmax(action_values).item()

    def save_memory_file(self):
        """
        Uloží pouze paměť zkušeností do souboru.
        """
        with open('/content/drive/MyDrive/Diplomka/model/DuelingDQN_V2/memory.pkl', 'wb') as f:
            pickle.dump(self.memory, f)

    def save_memory(self, state, action, reward, next_state, done):
        """
        Ukládá jednu zkušenost (transition) do replay bufferu.

        Args:
            state (np.ndarray): Aktuální stav.
            action (int): Prováděná akce.
            reward (float): Získaná odměna.
            next_state (np.ndarray): Následující stav.
            done (bool): Příznak, zda hra skončila.
        """
        self.memory.append((state, action, reward, next_state, done))

    def load_model(self, model_path='/content/drive/MyDrive/Diplomka/model/DuelingDQN_V2/q_network_complete.pt'):
        """
        Načte celý model včetně cílové sítě, optimizeru a replay bufferu.

        Args:
            model_path (str): Cesta k uloženému modelu.
        """
        model_state_dict = torch.load(model_path)

        # Load weights for the Q-network
        self.q_network['feature_layer'].load_state_dict(model_state_dict['feature_layer'])
        self.q_network['value_layer'].load_state_dict(model_state_dict['value_layer'])
        self.q_network['advantage_layer'].load_state_dict(model_state_dict['advantage_layer'])

        # Load weights for the target network
        self.target_network['feature_layer'].load_state_dict(model_state_dict['target_feature_layer'])
        self.target_network['value_layer'].load_state_dict(model_state_dict['target_value_layer'])
        self.target_network['advantage_layer'].load_state_dict(model_state_dict['target_advantage_layer'])

        # Load optimizer state if applicable
        self.optimizer.load_state_dict(model_state_dict['optimizer_state_dict'])

        # Load epsilon value if stored
        if 'epsilon' in model_state_dict:
            self.EPSILON = model_state_dict['epsilon']
        with open('/content/drive/MyDrive/Diplomka/model/DuelingDQN_V2/memory.pkl', 'rb') as f:
            self.memory = deque(pickle.load(f), maxlen=250000)
        print(len(self.memory))

        print("Model loaded successfully.")

    def train(self, batch_size):
        """
        Provede jeden krok učení pomocí minibatch vzorků ze zkušenostní paměti.

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

        predicted = self.forward(states_tensor, 'q_network')
        with torch.no_grad():
            next_predicted = self.forward(next_states_tensor, 'target_network')
            next_q_network_pred = self.forward(next_states_tensor, 'q_network')

        targets = predicted.clone().detach()
        actions = np.array([x[1] for x in minibatch])
        rewards = np.array([x[2] for x in minibatch])
        rewards_norm = (rewards - self.min_reward) / (self.max_reward - self.min_reward)
        dones = np.array([x[4] for x in minibatch])

        # Double DQN update: Use current Q-network to select best action, and target network for value
        best_actions = torch.argmax(next_q_network_pred, dim=1)
        target_values = next_predicted.gather(1, best_actions.unsqueeze(1)).squeeze(1)

        targets[range(batch_size), actions] = torch.FloatTensor(rewards).to(self.device) + self.GAMMA * target_values * (1 - torch.FloatTensor(dones).to(self.device))

        self.optimizer.zero_grad()
        loss = nn.MSELoss()(predicted, targets)
        loss.backward()
        self.optimizer.step()

        self.losses.append(loss.item())
        self.update_target_network()

    def update_target_network(self, tau=0.001):
        """
        Provádí tzv. soft update cílové sítě – postupné přibližování se k hlavní síti.

        Args:
            tau (float): Koeficient aktualizace (0 = žádná změna, 1 = úplné přepsání).
        """
        for layer_name in self.q_network.keys():
            for target_param, param in zip(self.target_network[layer_name].parameters(), self.q_network[layer_name].parameters()):
                target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

    def save_model(self):
        """
        Uloží všechny části modelu včetně target sítě, optimizeru a epsilonu.
        """
        torch.save({
            'feature_layer': self.q_network['feature_layer'].state_dict(),
            'value_layer': self.q_network['value_layer'].state_dict(),
            'advantage_layer': self.q_network['advantage_layer'].state_dict(),
            'target_feature_layer': self.target_network['feature_layer'].state_dict(),
            'target_value_layer': self.target_network['value_layer'].state_dict(),
            'target_advantage_layer': self.target_network['advantage_layer'].state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.EPSILON
        }, '/content/drive/MyDrive/Diplomka/model/DuelingDQN_V2/q_network_complete.pt')
        print("Model saved successfully.")
        with open('/content/drive/MyDrive/Diplomka/model/DuelingDQN_V2/memory.pkl', 'wb') as f:
            pickle.dump(self.memory, f)


    def update_learning_rate(self, new_learning_rate):
        """
        Aktualizuje hodnotu learning rate pro optimalizátor.

        Args:
            new_learning_rate (float): Nová hodnota učícího koeficientu.
        """
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_learning_rate

    def update_epsilon(self, new_epsilon):
        """
        Nastaví epsilon (pravděpodobnost náhodné akce).

        Args:
            new_epsilon (float): Nová hodnota epsilonu.
        """
        self.EPSILON = new_epsilon

    def epsilon_decay(self):
        """
        Snižuje epsilon podle přednastavené rychlosti (decay).
        """
        if self.EPSILON > self.EPSILON_MIN:
            self.EPSILON *= self.EPSILON_DECAY

    def save_model_checkpoint(self, round_count):
        """
        Uloží stav modelu jako checkpoint (pro konkrétní herní kolo).

        Args:
            round_count (int): Počet odehraných kol (slouží pro název složky).
        """
        checkpoint_dir = f'/content/drive/MyDrive/Diplomka/model/DuelingDQN_V2/checkpoint/{round_count}/'
        os.makedirs(checkpoint_dir, exist_ok=True)

        torch.save({
            'feature_layer': self.q_network['feature_layer'].state_dict(),
            'value_layer': self.q_network['value_layer'].state_dict(),
            'advantage_layer': self.q_network['advantage_layer'].state_dict(),
            'target_feature_layer': self.target_network['feature_layer'].state_dict(),
            'target_value_layer': self.target_network['value_layer'].state_dict(),
            'target_advantage_layer': self.target_network['advantage_layer'].state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.EPSILON
        }, f'{checkpoint_dir}q_network_complete.pt')

        # Uložení parametrů epsilon a paměti
        with open(f'{checkpoint_dir}model_params.pkl', 'wb') as f:
            pickle.dump({'epsilon': self.EPSILON}, f)
        with open(f'{checkpoint_dir}memory.pkl', 'wb') as f:
            pickle.dump(self.memory, f)

        print(f"Checkpoint saved successfully for round {round_count}.")