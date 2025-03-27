import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import time
import os
from collections import deque
import copy
import pickle  # ukládání a načítání souborů
from matplotlib import pyplot as plt

class Dueling_DQN(nn.Module):
    """
    Double Dueling Deep Q-Network (Dueling DQN) pro hru Blackjack.

    Tento agent využívá architekturu Double Dueling DQN, která odděluje odhad výhody (advantage) od hodnoty stavu (value).
    Trénuje se pomocí algoritmu Double DQN, kde se pro výběr akce používá hlavní síť a pro odhad její hodnoty cílová síť.
    """
    def __init__(self, state_size, action_size, device):
        """
        Inicializuje agenta a jeho parametry, sítě a normalizační rozsahy.

        Args:
            state_size (int): Počet vstupních vlastností.
            action_size (int): Počet možných akcí (např. hit, stand, double, split, insurance).
            device (torch.device): Použité zařízení (CPU/GPU).
        """
        super(Dueling_DQN, self).__init__()
        self.device = device
        self.state_size = state_size
        self.action_size = action_size
        self.columns_to_normalize = list(range(14))
        self.GAMMA = 0.99
        self.LR =  0.001
        self.losses = []
        self.EPSILON = 1
        self.EPSILON_MIN = 0.01
        self.EPSILON_DECAY = 0.99999
        self.memory = deque(maxlen=250000)

        # PyTorch model definition for Dueling DQN
        self.q_network = self.build_model().to(device)
        self.target_network = copy.deepcopy(self.q_network).to(device)
#        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.LR)
        self.optimizer = optim.SGD(self.q_network.parameters(), lr=self.LR, momentum=0.9)
        # Apply He Initialization
        self.apply_initialization()

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
        self.min_vals = np.array([self.mn_total, self.mn_d_show, self.mn_cards_dealt, self.mn_hilo_idx, self.mn_card_seen, self.mn_card_seen, self.mn_card_seen, self.mn_card_seen, self.mn_card_seen, self.mn_card_seen, self.mn_card_seen, self.mn_card_seen, self.mn_card_10_seen, self.mn_card_seen], dtype=float)
        self.max_vals = np.array([self.mx_total, self.mx_d_show, self.mx_cards_dealt, self.mx_hilo_idx, self.mx_card_seen, self.mx_card_seen, self.mx_card_seen, self.mx_card_seen, self.mx_card_seen, self.mx_card_seen, self.mx_card_seen, self.mx_card_seen, self.mx_card_10_seen, self.mx_card_seen], dtype=float)
        self.min_reward = np.array([-2.5], dtype=float)
        self.max_reward = np.array([2], dtype=float)

    def build_model(self):
        """
        Vytvoří architekturu Double Dueling DQN se samostatnými větvemi pro value a advantage.

        Returns:
            nn.ModuleDict: Struktura obsahující vrstvy sítě.
        """
        feature_layer = nn.Sequential(
            nn.Linear(self.state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )

        value_layer = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        advantage_layer = nn.Sequential(
            nn.Linear(128, 128),
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
        Inicializace vah vrstvy pomocí Kaiming uniformní metody.

        Args:
            m (nn.Module): Vrstva pro inicializaci.
        """
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def apply_initialization(self):
        """Inicializuje všechny vrstvy v Q i target síti."""

        for layer in self.q_network.values():
            layer.apply(self.initialize_weights)
        for layer in self.target_network.values():
            layer.apply(self.initialize_weights)

    def forward(self, x, network_type='q_network'):
        """
        Výpočet Q-hodnot pomocí sítě.

        Args:
            x (torch.Tensor): Vstupní stav.
            network_type (str): Typ sítě ('q_network' nebo 'target_network').

        Returns:
            torch.Tensor: Q hodnoty pro všechny akce.
        """
        if network_type == 'q_network':
            network = self.q_network
        elif network_type == 'target_network':
            network = self.target_network
        else:
            raise ValueError("Invalid network type. Choose 'q_network' or 'target_network'.")

        features = self.q_network['feature_layer'](x)
        value = self.q_network['value_layer'](features)
        advantage = self.q_network['advantage_layer'](features)
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values

    def normalize_states(self, states):
         """
        Normalizuje vstupy dle rozsahů.

        Args:
            states (np.ndarray): Pole vstupních stavů.

        Returns:
            np.ndarray: Normalizované stavy.
        """
        norm_states = states.copy().astype(float)
        norm_states[:, self.columns_to_normalize] = (states[:, self.columns_to_normalize] - self.min_vals) / (self.max_vals - self.min_vals)
        return norm_states

    def player_action(self, state, action_mask=None):
        """
        Vybere akci pomocí argmax nad Q hodnotami bez zohlednění průzkumu (bez epsilon).

        Args:
            state (np.ndarray): Vstupní stav.
            action_mask (np.ndarray, optional): Binární maska dostupných akcí.

        Returns:
            int: Index akce.
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
        Vybere akci pomocí epsilon-greedy strategie.

        Args:
            state (np.ndarray): Vstupní stav.
            action_mask (np.ndarray): Maska dostupných akcí.
            train (bool): Zda používat epsilon.

        Returns:
            int: Zvolená akce.
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
        """Uloží replay memory do souboru."""
        with open('/content/drive/MyDrive/Diplomka/model/DuelingDQN/memory.pkl', 'wb') as f:
            pickle.dump(self.memory, f)

    def save_memory(self, state, action, reward, next_state, done):
        """
        Uloží zkušenost do replay bufferu.

        Args:
            state: Aktuální stav.
            action: Provedená akce.
            reward: Získaná odměna.
            next_state: Nový stav po akci.
            done (bool): Zda hra skončila.
        """
        self.memory.append((state, action, reward, next_state, done))

    def load_model(self, model_path='/content/drive/MyDrive/Diplomka/model/DuelingDQN/q_network_complete.pt'):
        """
        Načte model včetně cílové sítě, optimalizátoru a paměti.

        Args:
            model_path (str): Cesta k souboru se stavem modelu.
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
        with open('/content/drive/MyDrive/Diplomka/model/DuelingDQN/memory.pkl', 'rb') as f:
            self.memory = deque(pickle.load(f), maxlen=250000)
        print(len(self.memory))

        print("Model loaded successfully.")

    def train(self, batch_size):
        """
        Provede trénink na základě náhodného batch ze zkušenostní paměti.

        Args:
            batch_size (int): Velikost minibatche.
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
        Soft update cílové sítě pomocí parametru tau.

        Args:
            tau (float): Váha směšování Q a target sítě.
        """
        for layer_name in self.q_network.keys():
            for target_param, param in zip(self.target_network[layer_name].parameters(), self.q_network[layer_name].parameters()):
                target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

    def save_model(self):
        """Uloží model a cílovou síť do souboru."""
        torch.save({
            'feature_layer': self.q_network['feature_layer'].state_dict(),
            'value_layer': self.q_network['value_layer'].state_dict(),
            'advantage_layer': self.q_network['advantage_layer'].state_dict(),
            'target_feature_layer': self.target_network['feature_layer'].state_dict(),
            'target_value_layer': self.target_network['value_layer'].state_dict(),
            'target_advantage_layer': self.target_network['advantage_layer'].state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.EPSILON
        }, '/content/drive/MyDrive/Diplomka/model/DuelingDQN/q_network_complete.pt')
        print("Model saved successfully.")

    def update_learning_rate(self, new_learning_rate):
        """
        Aktualizuje learning rate.

        Args:
            new_learning_rate (float): Nová hodnota learning rate.
        """
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_learning_rate

    def update_epsilon(self, new_epsilon):
        """
        Nastaví novou hodnotu epsilonu.

        Args:
            new_epsilon (float): Nová hodnota epsilon.
        """
        self.EPSILON = new_epsilon

    def epsilon_decay(self):
        """Aplikuje exponenciální decay na epsilon."""
        if self.EPSILON > self.EPSILON_MIN:
            self.EPSILON *= self.EPSILON_DECAY

    def save_model_checkpoint(self, round_count):
        """
        Uloží checkpoint modelu včetně vah, epsilonu a paměti.

        Args:
            round_count (int): Číslo aktuálního kola/hry.
        """
        checkpoint_dir = f'/content/drive/MyDrive/Diplomka/model/DuelingDQN/checkpoint/{round_count}/'
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