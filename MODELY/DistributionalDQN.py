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

class DistributionalDQN(nn.Module):
    """
    Distribuční Double DQN agent pro rozhodování v blackjacku.
    
    Implementuje distribuční Q-learning, kde výstupem není jen očekávaná hodnota, ale distribuce návratnosti. 
    Používá techniku Categorical DQN s měřením ztráty pomocí KL divergence a výpočtem distribuované cílové hodnoty (projekce na support atomy).
    
    Podporuje epsilon-greedy strategii, replay memory, ukládání, checkpointy a normalizaci vstupních stavů.
    """

    def __init__(self, state_size, action_size, device, num_atoms=51, v_min=-10, v_max=10):
        """
        Inicializuje instanci distribučního Double DQN agenta.
    
        Args:
            state_size (int): Počet vstupních proměnných reprezentujících herní stav.
            action_size (int): Počet možných akcí v prostředí.
            device (torch.device): Zařízení pro výpočty (např. torch.device("cuda") nebo "cpu").
            num_atoms (int): Počet diskrétních atomů reprezentujících distribuci návratnosti.
            v_min (float): Minimální hodnota návratnosti (support spodní mez).
            v_max (float): Maximální hodnota návratnosti (support horní mez).
        """
        super(DistributionalDQN, self).__init__()
        self.device = device
        self.state_size = state_size
        self.action_size = action_size
        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.delta_z = (v_max - v_min) / (num_atoms - 1)
        self.support = torch.linspace(v_min, v_max, num_atoms).to(device)
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

    def build_model(self):
        """
        Vytváří neuronovou síť s výstupem ve tvaru (action_size × num_atoms).
        
        Vrací:
            nn.Sequential: Sekvenční model.
        """
        return nn.Sequential(
            nn.Linear(self.state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, self.action_size * self.num_atoms)
        )

    def initialize_weights(self, m):
        """
        Inicializuje váhy vrstev pomocí Kaiming He inicializace vhodné pro ReLU aktivace.
    
        Args:
            m (torch.nn.Module): Vrstva, která má být inicializována.
        """
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x, network_type='q_network'):
        """
        Výpočet distribučních Q-hodnot.
    
        Args:
            x (torch.Tensor): Vstupní stav(y) tvaru [batch_size, state_size].
            network_type (str): 'q_network' nebo 'target_network'.
    
        Vrací:
            Tuple[torch.Tensor, torch.Tensor]: 
                - q_values (expected value pro každou akci), tvar [batch_size, action_size]
                - q_probs (pravděpodobnosti atomů), tvar [batch_size, action_size, num_atoms]
        """
        if network_type == 'q_network':
            network = self.q_network
        elif network_type == 'target_network':
            network = self.target_network
        else:
            raise ValueError("Invalid network type. Choose 'q_network' or 'target_network'.")

        q_atoms = self.q_network(x).view(-1, self.action_size, self.num_atoms)
        q_probs = torch.softmax(q_atoms, dim=2)  # Softmax over atoms
        q_values = torch.sum(q_probs * self.support, dim=2)  # Expected value
        return q_values, q_probs

    def normalize_states(self, states):
        """
        Normalizuje vstupy podle nastavených minim a maxim.
    
        Args:
            states (np.ndarray): Vstupní matice tvaru [N, state_size].
    
        Returns:
            np.ndarray: Normalizované stavy.
        """
        norm_states = states.copy().astype(float)
        norm_states[:, self.columns_to_normalize] = (states[:, self.columns_to_normalize] - self.min_vals) / (self.max_vals - self.min_vals)
        return norm_states

    def player_action(self, state, action_mask=None):
        """
        Vybere nejlepší akci na základě očekávaných Q-hodnot (bez epsilon-greedy).
    
        Args:
            state (np.ndarray): Vstupní stav.
            action_mask (np.ndarray, optional): Maska platných akcí.
    
        Returns:
            int: Index zvolené akce.
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
        Vybere akci pomocí epsilon-greedy strategie (během tréninku).
    
        Args:
            state (np.ndarray): Aktuální stav.
            action_mask (np.ndarray, optional): Maska platných akcí.
            train (bool): Pokud True, použije epsilon-greedy.
    
        Returns:
            int: Index akce.
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
        Trénuje síť pomocí vzorků z replay memory a KL divergence.
    
        Args:
            batch_size (int): Velikost trénovací minibatche.
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

        # Ztrátová funkce
        loss = F.kl_div(
            torch.log(predicted + 1e-8), projected_distribution, reduction='batchmean'
        )

        # Aktualizace parametrů
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Logování ztrát a aktualizace target network
        self.losses.append(loss.item())
        self.update_target_network()

    def project_distribution(self, target_probs, rewards, dones):
        """
        Projekce distribuované cílové hodnoty na fixní support atomy.
    
        Args:
            target_probs (Tensor): Pravděpodobnosti z target sítě.
            rewards (Tensor): Odměny.
            dones (Tensor): Flagy konce hry.
    
        Returns:
            Tensor: Projektovaná distribuční cílová pravděpodobnost.
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

        # Distribuce je rozdělena mezi sousední atomy
        proj_dist[batch_indices, l] += target_probs * (u - b)
        proj_dist[batch_indices, u] += target_probs * (b - l)

        return proj_dist

    def save_memory_file(self):
        """
        Uloží replay buffer (paměť zkušeností) na disk do souboru `memory.pkl`.
    
        Replay buffer obsahuje posloupnosti ve formátu (state, action, reward, next_state, done),
        které agent během učení nasbíral.
    
        Výstup:
            Soubor `memory.pkl` v definované cestě obsahující binárně serializovanou paměť.
        """
        with open('/content/drive/MyDrive/Diplomka/memory.pkl', 'wb') as f:
            pickle.dump(self.memory, f)

    def save_memory(self, state, action, reward, next_state, done):
        """Ukládá zkušenost do paměti."""
        self.memory.append((state, action, reward, next_state, done))

    def load_model(self):
        """Načte model, paměť a epsilon z disku."""
        self.q_network.load_state_dict(torch.load('/content/drive/MyDrive/Diplomka/model/DistDQN/q_network_complete.pt'))
        self.target_network.load_state_dict(torch.load('/content/drive/MyDrive/Diplomka/model/DistDQN/q_network_complete.pt'))
        with open('/content/drive/MyDrive/Diplomka/model/DistDQN/model_params.pkl', 'rb') as f:
            params = pickle.load(f)
            self.EPSILON = params['epsilon']
        with open('/content/drive/MyDrive/Diplomka/model/DistDQN/memory.pkl', 'rb') as f:
            self.memory = deque(pickle.load(f), maxlen=250000)
        print("Model and parameters loaded successfully.")

    def update_target_network(self, tau=0.005):
        """
        Soft update target sítě směrem k q síti.
    
        Args:
            tau (float): Koeficient pro interpolaci parametrů.
        """
        for target_param, param in zip(self.target_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

    def save_model(self):
        """Uloží stav modelu, epsilon a replay memory na disk."""
        torch.save(self.q_network.state_dict(), '/content/drive/MyDrive/Diplomka/model/DistDQN/q_network_complete.pt')
        with open('/content/drive/MyDrive/Diplomka/model/DistDQN/model_params.pkl', 'wb') as f:
            pickle.dump({'epsilon': self.EPSILON}, f)
        with open('/content/drive/MyDrive/Diplomka/model/DistDQN/memory.pkl', 'wb') as f:
            pickle.dump(self.memory, f)

    def update_learning_rate(self, new_learning_rate):
        """
        Nastaví novou learning rate pro optimizer.
    
        Args:
            new_learning_rate (float): Nová hodnota learning rate.
        """
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_learning_rate

    def update_epsilon(self, new_epsilon):
        """
        Nastaví novou hodnotu epsilon.
    
        Args:
            new_epsilon (float): Nová hodnota epsilon.
        """
        self.EPSILON = new_epsilon

    def epsilon_decay(self):
        """
        Aplikuje decay na epsilon (pokud není na minimu).
        """
        if self.EPSILON > self.EPSILON_MIN:
            self.EPSILON *= self.EPSILON_DECAY

    def save_model_checkpoint(self, round_count):
        """
        Uloží checkpoint modelu pro daný počet odehraných her (kol).
    
        Args:
            round_count (int): Počet odehraných kol (pro název složky).
        """
        checkpoint_dir = f'/content/drive/MyDrive/Diplomka/model/DistDQN/checkpoint/{round_count}/'
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