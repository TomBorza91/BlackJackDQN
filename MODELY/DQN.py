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
import BlackJack
from matplotlib import pyplot as plt

class DQN(nn.Module):
    """
    Agent pro hru BlackJack založený na klasickém Double DQN algoritmu.

    Využívá dvě neuronové sítě (hlavní a cílovou) a paměť zkušeností (experience replay),
    díky čemuž se učí z minulých situací a iterativně se přizpůsobuje.

    Vstupem je vektor s 17 hodnotami reprezentujícími herní stav, včetně hodnoty ruky,
    Hi-Lo indexu, počtu rozdáných a viděných karet. Výstupem je očekávaná hodnota pro každou možnou akci.
    """
    def __init__(self, state_size, action_size, device):
        """
        Inicializace modelu DQN.

        Args:
            state_size (int): Počet vstupních prvků reprezentujících stav.
            action_size (int): Počet možných akcí v prostředí.
            device (str): Výpočetní zařízení ("cpu" nebo "cuda").
        """
        super(DQN, self).__init__()
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

        # PyTorch model definition
        self.q_network = self.build_model().to(device)
        self.q_network.apply(self.initialize_weights)  # Aplikace inicializace vah
        self.target_network = copy.deepcopy(self.q_network).to(device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.LR)

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

    def initialize_weights(self, m):
        """
        Inicializuje váhy sítě pomocí He inicializace.
        """
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def build_model(self):
        """
        Definuje architekturu neuronové sítě.

        Returns:
            nn.Sequential: Sekvenční síť se dvěma skrytými vrstvami.
        """
        return nn.Sequential(
            nn.Linear(self.state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, self.action_size)
        )

    def forward(self, x):
        """
        Provede forward pass přes hlavní síť.

        Args:
            x (torch.Tensor): Vstupní tenzor.

        Returns:
            torch.Tensor: Výstupy pro jednotlivé akce.
        """
        return self.q_network(x)

    def normalize_states(self, states):
        """
        Normalizuje vstupní stavy podle předdefinovaných rozsahů.

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
        Vrací akci s nejvyšší Q hodnotou (bez epsilon-greedy průzkumu).

        Args:
            state (np.ndarray): Vstupní stav.
            action_mask (np.ndarray, optional): Maska legálních akcí.

        Returns:
            int: Index vybrané akce.
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
        Vybere akci pomocí epsilon-greedy strategie.

        Args:
            state (np.ndarray): Vstupní stav.
            action_mask (np.ndarray, optional): Maska platných akcí.
            train (bool): True, pokud jsme v trénovací fázi.

        Returns:
            int: Vybraná akce.
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
        with open('/content/drive/MyDrive/Diplomka/model/DoubleDQN/memory.pkl', 'wb') as f:
            pickle.dump(self.memory, f)

    def save_memory(self, state, action, reward, next_state, done):
        """
        Uloží jednu zkušenost do paměti.

        Args:
            state (np.ndarray)
            action (int)
            reward (float)
            next_state (np.ndarray)
            done (bool)
        """
        self.memory.append((state, action, reward, next_state, done))

    def load_model(self):
        """Načte model, epsilon a replay paměť ze souboru."""
        self.q_network.load_state_dict(torch.load('/content/drive/MyDrive/Diplomka/model/DoubleDQN/q_network_complete.pt'))
        self.target_network.load_state_dict(torch.load('/content/drive/MyDrive/Diplomka/model/DoubleDQN/q_network_complete.pt'))
        with open('/content/drive/MyDrive/Diplomka/model/DoubleDQN/model_params.pkl', 'rb') as f:
            params = pickle.load(f)
            self.EPSILON = params['epsilon']
        with open('/content/drive/MyDrive/Diplomka/model/DoubleDQN/memory.pkl', 'rb') as f:
            self.memory = deque(pickle.load(f), maxlen=250000)
        print("Model and parameters loaded successfully.")

    def train(self, batch_size):
        """
        Trénuje model na náhodné minibatchi ze zkušeností v paměti.

        Args:
            batch_size (int): Velikost trénovací dávky.
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
        Soft update parametrů cílové sítě podle hlavní sítě.

        Args:
            tau (float): Váha aktualizace.
        """
        for target_param, param in zip(self.target_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

    def save_model(self):
         """Uloží model, epsilon a paměť do souboru."""
        torch.save(self.q_network.state_dict(), '/content/drive/MyDrive/Diplomka/model/DoubleDQN/q_network_complete.pt')
        with open('/content/drive/MyDrive/Diplomka/model/DoubleDQN/model_params.pkl', 'wb') as f:
            pickle.dump({'epsilon': self.EPSILON}, f)
        with open('/content/drive/MyDrive/Diplomka/model/DoubleDQN/memory.pkl', 'wb') as f:
            pickle.dump(self.memory, f)

    def update_learning_rate(self, new_learning_rate):
        """
        Aktualizuje učící rychlost pro optimalizátor.

        Args:
            new_learning_rate (float): Nová hodnota learning rate.
        """
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_learning_rate

    def update_epsilon(self, new_epsilon):
        """
        Nastaví novou hodnotu epsilonu.

        Args:
            new_epsilon (float)
        """
        self.EPSILON = new_epsilon

    def epsilon_decay(self):
        """Postupně snižuje hodnotu epsilonu podle decay faktoru."""
        if self.EPSILON > self.EPSILON_MIN:
            self.EPSILON *= self.EPSILON_DECAY

    def save_model_checkpoint(self, round_count):
        """
        Uloží checkpoint modelu do specifické složky podle počtu kol.

        Args:
            round_count (int): Číslo aktuálního kola.
        """        checkpoint_dir = f'/content/drive/MyDrive/Diplomka/model/DoubleDQN/checkpoint/{round_count}/'
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