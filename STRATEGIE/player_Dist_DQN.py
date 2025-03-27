import DistributionalDQN
import numpy as np
import pickle
import time

class PlayerDistDQN:
    """
    Hráč využívající distribuční DQN model (C51) pro rozhodování v BlackJacku.

    Distribuční DQN (C51) modeluje celé rozdělení budoucí návratnosti (nikoliv jen očekávanou hodnotu)
    a na základě něj volí nejlepší akci. Tato třída zahrnuje základní ošetření nepovolených akcí pomocí masky.
    """
    def __init__(self, device):
        """
        Inicializuje hráče a načte C51 model.

        Args:
            device (str): Výpočetní zařízení – 'cpu' nebo 'cuda'.
        """
        self.device = device
        self.agent = self.__load_model(17, 5)
        #self.stats = self.load_stats()
        self.dqn_moves = {
            0:"HIT",
            1:"STAND",
            2:"DOUBLE",
            3:"SPLIT",
            4:"INSURANCE"
        }

    def __load_model(self, i, o):
        """
        Načte instanci distribučního DQN modelu (C51) s daným počtem vstupů a výstupů.

        Args:
            i (int): Počet vstupních prvků (features).
            o (int): Počet výstupních akcí.

        Returns:
            DistributionalDQN.DistributionalDQN: Natrénovaný agent s metodou player_action().
        """
        agent = DistributionalDQN.DistributionalDQN(i, o, self.device)
        agent.load_model()
        return agent
        
    def get_bet(self, i_idx, i_min=1, i_max=10):
        """
        Vrací výši sázky. V této verzi je vždy rovna minimální sázce.

        Args:
            i_idx (int): Aktuální Hi-Lo index (nepoužívá se).
            i_min (int): Minimální možná sázka.
            i_max (int): Maximální možná sázka (nepoužívá se).

        Returns:
            int: Výše sázky.
        """
        return i_min
    
    
    def get_move(self, i_round_data):
        """
        Vrací vybranou akci na základě aktuální herní situace.
        Používá distribuční DQN model (C51) a aplikuje masku pro validní akce.

        Args:
            i_round_data (dict): Slovník s informacemi o aktuálním stavu hry.
                Obsahuje:
                    - "total" (int): Celková hodnota hráčovy ruky.
                    - "d_show" (int): Viditelná karta dealera.
                    - "cards_dealt" (int): Počet dosud rozdaných karet.
                    - "hilo_idx" (int): Aktuální Hi-Lo index.
                    - "cards_seen_cnt" (dict[int, int]): Výskyt jednotlivých hodnot karet.
                    - "soft" (int): 1 pokud je ruka měkká (obsahuje eso jako 11), jinak 0.
                    - "pair" (bool): True pokud má hráč pár.
                    - "init_deal" (int): 1 pokud se jedná o první rozdání.

        Returns:
            str: Vybraná akce podle modelu.
                Možné hodnoty: "HIT", "STAND", "DOUBLE", "SPLIT", "INSURANCE"
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
