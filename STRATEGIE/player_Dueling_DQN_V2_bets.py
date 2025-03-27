import Dueling_DQN_V2
import numpy as np
import pickle
import time

class PlayerDuelingDQN_V2_Bets:
    """
    Hybridní hráč pro BlackJack využívající pokročilý Double Dueling DQN model
    a dynamické sázení na základě Hi-Lo indexu.

    Model je trénován s rozšířeným vstupem (27 feature) a zohledňuje složení balíčku
    i relativní hustotu nízkých a vysokých karet. Sázení je adaptivní podle síly balíčku.

    Použití: simulace hráče se schopností pokročilé predikce + výhodné sázkové politiky.
    """
    def __init__(self, device):
         """
        Inicializuje hráče a načítá pokročilý Double Dueling DQN model.

        Args:
            device (str): Zařízení pro výpočty ('cpu' nebo 'cuda').
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
        Interní metoda pro vytvoření a načtení Double Dueling DQN modelu.

        Args:
            i (int): Počet vstupních neuronů.
            o (int): Počet možných akcí.

        Returns:
            Dueling_DQN_V2.Dueling_DQN_V2: Inicializovaný natrénovaný agent.
        """
        agent = Dueling_DQN_V2.Dueling_DQN_V2(i, o, self.device)
        agent.load_model()
        return agent
        
    def get_bet(self, i_idx, i_min=1, i_max=10):
        """
        Dynamicky určuje výši sázky na základě aktuální hodnoty Hi-Lo indexu.

        Args:
            i_idx (int): Hodnota Hi-Lo indexu.
            i_min (int): Minimální povolená sázka.
            i_max (int): Maximální sázka (aktuálně neomezeno).

        Returns:
            float: Výše sázky jako násobek minimální podle výhodnosti balíčku.
        """
        # aby nedospěli k podezření
        if i_idx > 10:
            idx = 5
        # nemůžu vsadit méně než min bet
        elif i_idx < 2:
            idx = 1
        else: 
            idx = i_idx/2
        return i_min*idx
    
    
    def get_move(self, i_round_data):
       """
        Rozhoduje o herním tahu pomocí pokročilého Dueling DQN modelu s rozšířeným vstupem.

        Vstupní vektor obsahuje statistiku o počtu viděných karet, počtech high/low karet,
        jejich poměr, zbylé karty v balíčku, hodnotu ruky, přítomnost páru nebo soft hand atd.

        Maskuje nedostupné akce podle pravidel a model predikuje akci s nejvyšší hodnotou.

        Args:
            i_round_data (dict): Informace o stavu aktuálního kola:
                - 'total' (int): Hodnota hráčovy ruky.
                - 'd_show' (int): Viditelná karta dealera.
                - 'cards_dealt' (int): Počet již rozdaných karet.
                - 'hilo_idx' (int): Aktuální hodnota Hi-Lo indexu.
                - 'cards_seen_cnt' (dict[int, int]): Počet viděných karet dle hodnoty (2–11).
                - 'cards' (list[int]): Karty hráče.
                - 'soft' (int): 1 pokud má hráč soft hand.
                - 'pair' (bool): True pokud má hráč pár.
                - 'init_deal' (int): 1 pokud se jedná o první rozdání.

        Returns:
            str: Název akce – {"HIT", "STAND", "DOUBLE", "SPLIT", "INSURANCE"}.
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
