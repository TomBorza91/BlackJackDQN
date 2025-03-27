import Dueling_DQN
import numpy as np
import pickle
import time

class PlayerDuelingDQNBets:
    """
    Hybridní hráč pro BlackJack, který kombinuje Double Dueling DQN model pro rozhodování
    s variabilní výší sázky dle Hi-Lo indexu.

    Cílem této strategie je maximalizovat zisk:
    - pomocí přesného výběru akcí přes RL model (Dueling DQN),
    - a dynamickým navyšováním sázky při výhodném složení balíčku.

    Vstupní vektor pro model obsahuje 17 prvků reprezentujících stav hry.
    """
    def __init__(self, device):
        """
        Inicializuje hráče a načítá natrénovaný Double Dueling DQN model.

        Args:
            device (str): Výpočetní zařízení ('cpu' nebo 'cuda').
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
        Interní metoda pro vytvoření a načtení Double Dueling DQN modelu.

        Args:
            i (int): Počet vstupních neuronů (features).
            o (int): Počet výstupních akcí.

        Returns:
            Dueling_DQN.Dueling_DQN: Načtený agent s natrénovanými váhami.
        """
        agent = Dueling_DQN.Dueling_DQN(i, o, self.device)
        agent.load_model()
        return agent
        
    def get_bet(self, i_idx, i_min=1, i_max=10):
        """
        Dynamicky určuje sázku podle hodnoty Hi-Lo indexu.

        Vyšší Hi-Lo index → vyšší sázka (ale nikdy méně než minimum).
        Návratová hodnota je násobkem minimální sázky.

        Args:
            i_idx (int): Aktuální hodnota Hi-Lo indexu.
            i_min (int): Minimální povolená sázka.
            i_max (int): Maximální povolená sázka (aktuálně neomezeno).

        Returns:
            float: Výše sázky.
        """
        if i_idx > 10:
            idx = 5
        elif i_idx < 2:
            idx = 1
        else: 
            idx = i_idx/2
        return i_min*idx
    
    def get_move(self, i_round_data):
        """
        Rozhoduje o tahu na základě stavu hry pomocí Double Dueling DQN modelu.

        Vytváří vstupní vektor s 17 prvky, generuje masku možných akcí dle pravidel,
        a vybírá akci s nejvyšší očekávanou hodnotou.

        Args:
            i_round_data (dict): Slovník se stavem kola:
                - 'total' (int): Hodnota ruky hráče.
                - 'd_show' (int): Viditelná karta dealera.
                - 'cards_dealt' (int): Počet rozdaných karet.
                - 'hilo_idx' (int): Aktuální Hi-Lo index.
                - 'cards_seen_cnt' (dict[int, int]): Počet již viděných karet 2–11.
                - 'soft' (int): 1 pokud ruka obsahuje eso jako 11.
                - 'pair' (bool): True pokud má hráč pár.
                - 'init_deal' (int): 1 pokud se jedná o počáteční rozdání.

        Returns:
            str: Vybraná akce – jedna z {"HIT", "STAND", "DOUBLE", "SPLIT", "INSURANCE"}.
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
