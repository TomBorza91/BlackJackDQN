import Dueling_DQN
import numpy as np
import pickle
import time

class PlayerDuelingDQN:
    """
    Hráč používající Double Dueling DQN model (deep reinforcement learning) k rozhodování
    o akcích během hry BlackJack.

    Model vyhodnocuje aktuální stav hry jako vektor 17 prvků a rozhoduje pomocí neuronové sítě.
    """
    def __init__(self, device):
        """
        Inicializuje hráče s Double Dueling DQN agentem.

        Args:
            device (str): Zařízení pro výpočty ('cpu' nebo 'cuda').
        """
        self.device = device
        self.agent = self.__load_model(17, 5)
        self.dqn_moves = {
            0:"HIT",
            1:"STAND",
            2:"DOUBLE",
            3:"SPLIT",
            4:"INSURANCE"
        }

    def __load_model(self, i, o):
        """
        Načte model Double Dueling DQN s daným počtem vstupů a výstupů.

        Args:
            i (int): Počet vstupních prvků (features).
            o (int): Počet možných akcí.

        Returns:
            Dueling_DQN.Dueling_DQN: Načtený model agenta.
        """
        agent = Dueling_DQN.Dueling_DQN(i, o, self.device)
        agent.load_model()
        return agent
        
    def get_bet(self, i_idx, i_min=1, i_max=10):
        """
        Vrací výši sázky. V této verzi vrací vždy minimální hodnotu.

        Args:
            i_idx (int): Hi-Lo index (nevyužívá se).
            i_min (int): Minimální sázka.
            i_max (int): Maximální sázka (nevyužívá se).

        Returns:
            int: Výše sázky (vždy i_min).
        """
        return i_min
    
    
    def get_move(self, i_round_data):
        """
        Vrací akci hráče na základě aktuální situace ve hře, vyhodnocenou pomocí Double Dueling DQN.

        Args:
            i_round_data (dict): Slovník obsahující popis aktuální herní situace.
                Klíče:
                    - "total" (int): Hodnota hráčovy ruky
                    - "d_show" (int): Dealerova viditelná karta
                    - "cards_dealt" (int): Počet rozdaných karet
                    - "hilo_idx" (int): Aktuální Hi-Lo index
                    - "cards_seen_cnt" (dict[int, int]): Počty vidění jednotlivých karet
                    - "soft" (int): 1, pokud je ruka měkká (soft hand), jinak 0
                    - "pair" (bool): True, pokud má hráč pár
                    - "init_deal" (int): 1, pokud se jedná o první rozdání

        Returns:
            str: Akce vybraná agentem.
                Jedna z: "HIT", "STAND", "DOUBLE", "SPLIT", "INSURANCE"
        """
        observation = [
            i_round_data["total"], 
            i_round_data["d_show"], 
            i_round_data["cards_dealt"], 
            i_round_data['hilo_idx'], 
            i_round_data["cards_seen_cnt"][2], 
            i_round_data["cards_seen_cnt"][3], 
            i_round_data["cards_seen_cnt"][4], 
            i_round_data["cards_seen_cnt"][5], 
            i_round_data["cards_seen_cnt"][6], 
            i_round_data["cards_seen_cnt"][7], 
            i_round_data["cards_seen_cnt"][8], 
            i_round_data["cards_seen_cnt"][9], 
            i_round_data["cards_seen_cnt"][10], 
            i_round_data["cards_seen_cnt"][11], 
            i_round_data["soft"],
            i_round_data["pair"], 
            i_round_data["init_deal"]
        ]
 
        if i_round_data["total"] >= 20:
            if i_round_data["init_deal"]:
                if i_round_data["total"] == 21:
                    action_mask = np.array([0,1,0,0,0])
                if i_round_data["total"] == 20:
                    action_mask = np.array([0,1,0,0,1])
            else:
                action_mask = np.array([0,1,0,0,0])
        else:
            action_mask = np.array([1,1,1,1,1])
        if not i_round_data["pair"]:
            action_mask[3] = 0
        if not i_round_data["init_deal"] or (i_round_data["init_deal"] and (i_round_data["d_show"] != 11 or  i_round_data['hilo_idx'] < 3)):
            action_mask[4] = 0 
    
        action = self.agent.player_action(np.reshape(observation, [1,17]), action_mask)
    
        if (action == 3 and not i_round_data["pair"]) or (action == 4 and not i_round_data["init_deal"]):
            print('FAUL!')
            print(action)
            if i_round_data["total"] > 16:
                action = 1
            else:
                action = 0
    
        return self.dqn_moves[action]
