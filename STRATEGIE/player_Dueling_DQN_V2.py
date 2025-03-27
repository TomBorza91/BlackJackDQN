import Dueling_DQN_V2
import numpy as np
import pickle
import time

class PlayerDuelingDQN_V2:
    """
    Hráč využívající pokročilý Double Dueling DQN model pro rozhodování v BlackJacku.

    Model pracuje s rozšířeným 27-prvkovým vstupem zahrnujícím:
    - surová data o kartách
    - agregované součty nízkých, středních a vysokých karet
    - poměry mezi typy karet
    - odhadované informace o riziku (např. kolik chybí do 21)

    Tento hráč provádí rozhodnutí mezi akcemi HIT, STAND, DOUBLE, SPLIT, INSURANCE
    na základě výstupu modelu a aplikuje masky k omezení nepřípustných akcí.
    """
    def __init__(self, device):
        """
        Inicializuje hráče a načte trénovaný model Double Dueling DQN V2.

        Args:
            device (str): Výpočetní zařízení ('cpu' nebo 'cuda').
        """
        self.device = device
        self.agent = self.__load_model(27, 5)
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
        Načte model Dueling DQN V2 s daným počtem vstupů a výstupů.

        Args:
            i (int): Počet vstupních neuronů (features).
            o (int): Počet možných akcí.

        Returns:
            Dueling_DQN_V2.Dueling_DQN_V2: Natrénovaný model agenta.
        """
        agent = Dueling_DQN_V2.Dueling_DQN_V2(i, o, self.device)
        agent.load_model()
        return agent
        
    def get_bet(self, i_idx, i_min=1, i_max=10):
        """
        Vrací výši sázky. Momentálně vždy minimální hodnota.

        Args:
            i_idx (int): Hi-Lo index (momentálně se nevyužívá).
            i_min (int): Minimální možná sázka.
            i_max (int): Maximální možná sázka.

        Returns:
            int: Výše sázky.
        """
    return i_min
    
    
    def get_move(self, i_round_data):
        """
        Na základě herního stavu a masky možných tahů vrací akci hráče podle modelu.

        Args:
            i_round_data (dict): Slovník s popisem aktuální situace.
                Obsahuje:
                    - "total" (int): Součet hráčovy ruky
                    - "d_show" (int): Dealerova viditelná karta
                    - "cards_dealt" (int): Počet rozdaných karet
                    - "hilo_idx" (int): Aktuální Hi-Lo index
                    - "cards_seen_cnt" (dict[int, int]): Počty výskytu jednotlivých hodnot karet
                    - "soft" (int): 1 pokud je ruka měkká, jinak 0
                    - "pair" (bool): True, pokud má hráč pár
                    - "init_deal" (int): 1 pokud je to první rozdání
                    - "cards" (list[int]): Seznam hráčových karet

        Returns:
            str: Vybraná akce.
                Možnosti: "HIT", "STAND", "DOUBLE", "SPLIT", "INSURANCE"
        """
        lo_cards=[i_round_data["cards_seen_cnt"][2], i_round_data["cards_seen_cnt"][3], i_round_data["cards_seen_cnt"][4], i_round_data["cards_seen_cnt"][5], i_round_data["cards_seen_cnt"][6]]
        mid_cards = [i_round_data["cards_seen_cnt"][7], i_round_data["cards_seen_cnt"][8], i_round_data["cards_seen_cnt"][9]]
        hi_cards = [i_round_data["cards_seen_cnt"][10], i_round_data["cards_seen_cnt"][11]]
        split_total = 0
        if i_round_data["pair"]:
            if 11 in i_round_data["cards"]:
                split_total = 11
            else:
                split_total = i_round_data["total"]/2
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
            sum(lo_cards),
            sum(mid_cards),
            sum(hi_cards),
            sum(hi_cards) / sum(lo_cards) if sum(lo_cards) >= 1 else sum(hi_cards),
            sum(lo_cards)/(416-i_round_data["cards_dealt"]),
            sum(mid_cards)/(416-i_round_data["cards_dealt"]),
            sum(hi_cards)/(416-i_round_data["cards_dealt"]),
            416-i_round_data["cards_dealt"],
            21 - i_round_data["total"] if i_round_data["total"] <= 21 else 21,
            split_total,
            i_round_data["soft"],
            i_round_data["pair"],
            i_round_data["init_deal"]]
 
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
    
        action = self.agent.player_action(np.reshape(observation, [1,27]), action_mask)
    
        if (action == 3 and not i_round_data["pair"]) or (action == 4 and not i_round_data["init_deal"]):
            print('FAUL!')
            print(action)
            if i_round_data["total"] > 16:
                action = 1
            else:
                action = 0
    
        return self.dqn_moves[action]
