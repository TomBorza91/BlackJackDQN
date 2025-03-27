class PlayerBS:
    """
    Implementace základní strategie pro hru BlackJack (Basic Strategy, BS).
    """
    def __init__(self):
        """
        Inicializuje hráče se základní strategií.
        """
        self.strategy = self.get_strategy()

    def get_strategy(self):
        """
        Vrací předdefinovanou základní strategii pro všechny možné situace hráče.

        Returns:
            dict: Vnořený slovník obsahující pravidla pro akce (HIT, STAND, DOUBLE, SPLIT)
                  podle kombinace hráčových a dealerových karet.
        """

        strategy_data = {
            "HIT": {
                4: {
                    11: "HIT",
                    2: "HIT",
                    3: "HIT",
                    4: "HIT",
                    5: "HIT",
                    6: "HIT",
                    7: "HIT",
                    8: "HIT",
                    9: "HIT",
                    10: "HIT",
                },
                5: {
                    11: "HIT",
                    2: "HIT",
                    3: "HIT",
                    4: "HIT",
                    5: "HIT",
                    6: "HIT",
                    7: "HIT",
                    8: "HIT",
                    9: "HIT",
                    10: "HIT",
                },
                6: {
                    11: "HIT",
                    2: "HIT",
                    3: "HIT",
                    4: "HIT",
                    5: "HIT",
                    6: "HIT",
                    7: "HIT",
                    8: "HIT",
                    9: "HIT",
                    10: "HIT",
                },
                7: {
                    11: "HIT",
                    2: "HIT",
                    3: "HIT",
                    4: "HIT",
                    5: "HIT",
                    6: "HIT",
                    7: "HIT",
                    8: "HIT",
                    9: "HIT",
                    10: "HIT",
                },
                8: {
                    11: "HIT",
                    2: "HIT",
                    3: "HIT",
                    4: "HIT",
                    5: "HIT",
                    6: "HIT",
                    7: "HIT",
                    8: "HIT",
                    9: "HIT",
                    10: "HIT",
                },
                9: {
                    11: "HIT",
                    2: "HIT",
                    3: "HIT",
                    4: "HIT",
                    5: "HIT",
                    6: "HIT",
                    7: "HIT",
                    8: "HIT",
                    9: "HIT",
                    10: "HIT",
                },
                10: {
                    11: "HIT",
                    2: "HIT",
                    3: "HIT",
                    4: "HIT",
                    5: "HIT",
                    6: "HIT",
                    7: "HIT",
                    8: "HIT",
                    9: "HIT",
                    10: "HIT",
                },
                11: {
                    11: "HIT",
                    2: "HIT",
                    3: "HIT",
                    4: "HIT",
                    5: "HIT",
                    6: "HIT",
                    7: "HIT",
                    8: "HIT",
                    9: "HIT",
                    10: "HIT",
                },
                12: {
                    11: "HIT",
                    2: "HIT",
                    3: "HIT",
                    4: "STAND",
                    5: "STAND",
                    6: "STAND",
                    7: "HIT",
                    8: "HIT",
                    9: "HIT",
                    10: "HIT",
                },
                13: {
                    11: "HIT",
                    2: "STAND",
                    3: "STAND",
                    4: "STAND",
                    5: "STAND",
                    6: "STAND",
                    7: "HIT",
                    8: "HIT",
                    9: "HIT",
                    10: "HIT",
                },
                14: {
                    11: "HIT",
                    2: "STAND",
                    3: "STAND",
                    4: "STAND",
                    5: "STAND",
                    6: "STAND",
                    7: "HIT",
                    8: "HIT",
                    9: "HIT",
                    10: "HIT",
                },
                15: {
                    11: "HIT",
                    2: "STAND",
                    3: "STAND",
                    4: "STAND",
                    5: "STAND",
                    6: "STAND",
                    7: "HIT",
                    8: "HIT",
                    9: "HIT",
                    10: "HIT",
                },
                16: {
                    11: "HIT",
                    2: "STAND",
                    3: "STAND",
                    4: "STAND",
                    5: "STAND",
                    6: "STAND",
                    7: "HIT",
                    8: "HIT",
                    9: "HIT",
                    10: "HIT",
                },
                17: {
                    11: "STAND",
                    2: "STAND",
                    3: "STAND",
                    4: "STAND",
                    5: "STAND",
                    6: "STAND",
                    7: "STAND",
                    8: "STAND",
                    9: "STAND",
                    10: "STAND",
                },
                18: {
                    11: "STAND",
                    2: "STAND",
                    3: "STAND",
                    4: "STAND",
                    5: "STAND",
                    6: "STAND",
                    7: "STAND",
                    8: "STAND",
                    9: "STAND",
                    10: "STAND",
                },
                19: {
                    11: "STAND",
                    2: "STAND",
                    3: "STAND",
                    4: "STAND",
                    5: "STAND",
                    6: "STAND",
                    7: "STAND",
                    8: "STAND",
                    9: "STAND",
                    10: "STAND",
                },
                20: {
                    11: "STAND",
                    2: "STAND",
                    3: "STAND",
                    4: "STAND",
                    5: "STAND",
                    6: "STAND",
                    7: "STAND",
                    8: "STAND",
                    9: "STAND",
                    10: "STAND",
                },
                21: {
                    11: "STAND",
                    2: "STAND",
                    3: "STAND",
                    4: "STAND",
                    5: "STAND",
                    6: "STAND",
                    7: "STAND",
                    8: "STAND",
                    9: "STAND",
                    10: "STAND",
                },
            },
            "HIT_S": {
                18: {
                    11: "STAND",
                    2: "STAND",
                    3: "STAND",
                    4: "STAND",
                    5: "STAND",
                    6: "STAND",
                    7: "STAND",
                    8: "STAND",
                    9: "HIT",
                    10: "HIT",
                },
                19: {
                    11: "HIT",
                    2: "HIT",
                    3: "HIT",
                    4: "HIT",
                    5: "HIT",
                    6: "HIT",
                    7: "HIT",
                    8: "HIT",
                    9: "STAND",
                    10: "STAND",
                },
            },
            "DOUBLE": {
                9: {2: "DOUBLE", 3: "DOUBLE", 4: "DOUBLE", 5: "DOUBLE"},
                10: {
                    2: "DOUBLE",
                    3: "DOUBLE",
                    4: "DOUBLE",
                    5: "DOUBLE",
                    6: "DOUBLE",
                    7: "DOUBLE",
                    8: "DOUBLE",
                    9: "DOUBLE",
                },
                11: {
                    2: "DOUBLE",
                    3: "DOUBLE",
                    4: "DOUBLE",
                    5: "DOUBLE",
                    6: "DOUBLE",
                    7: "DOUBLE",
                    8: "DOUBLE",
                    9: "DOUBLE",
                    10: "DOUBLE",
                },
            },
            "DOUBLE_S": {
                12: {5: "DOUBLE"},
                13: {5: "DOUBLE", 6: "DOUBLE"},
                14: {5: "DOUBLE", 6: "DOUBLE"},
                15: {5: "DOUBLE", 6: "DOUBLE"},
                16: {5: "DOUBLE", 6: "DOUBLE"},
                17: {3: "DOUBLE", 4: "DOUBLE", 5: "DOUBLE", 6: "DOUBLE"},
                18: {4: "DOUBLE", 5: "DOUBLE", 6: "DOUBLE"},
            },
            "SPLIT": {
                11: {
                    11: "SPLIT",
                    2: "SPLIT",
                    3: "SPLIT",
                    4: "SPLIT",
                    5: "SPLIT",
                    6: "SPLIT",
                    7: "SPLIT",
                    8: "SPLIT",
                    9: "SPLIT",
                    10: "SPLIT",
                },
                2: {2: "SPLIT", 3: "SPLIT", 4: "SPLIT", 5: "SPLIT", 6: "SPLIT", 7: "SPLIT"},
                3: {2: "SPLIT", 3: "SPLIT", 4: "SPLIT", 5: "SPLIT", 6: "SPLIT", 7: "SPLIT"},
                6: {2: "SPLIT", 3: "SPLIT", 4: "SPLIT", 5: "SPLIT", 6: "SPLIT", 7: "SPLIT"},
                7: {2: "SPLIT", 3: "SPLIT", 4: "SPLIT", 5: "SPLIT", 6: "SPLIT", 7: "SPLIT"},
                8: {
                    11: "SPLIT",
                    2: "SPLIT",
                    3: "SPLIT",
                    4: "SPLIT",
                    5: "SPLIT",
                    6: "SPLIT",
                    7: "SPLIT",
                    8: "SPLIT",
                    9: "SPLIT",
                    10: "SPLIT",
                },
                9: {
                    2: "SPLIT",
                    3: "SPLIT",
                    4: "SPLIT",
                    5: "SPLIT",
                    6: "SPLIT",
                    8: "SPLIT",
                    9: "SPLIT",
                },
            },
        }
        return strategy_data


    def get_bet(self, i_idx, i_min=1, i_max=10):
        """
        Vrací výši sázky. V této implementaci vždy vrací minimální sázku.

        Args:
            i_idx (int): Hodnota Hi-Lo indexu (momentálně se nevyužívá).
            i_min (int): Minimální možná sázka.
            i_max (int): Maximální možná sázka.

        Returns:
            int: Výše sázky (vždy i_min).
        """
        print('get_bet:', i_min)
        return i_min
    
    
    def get_move(self, i_round_data):
        """
        Vrací akci hráče na základě aktuální situace podle základní strategie.

        Args:
            i_round_data (dict): Slovník popisující stav ruky a viditelnou kartu dealera.
                Klíče:
                    - "d_show" (int): Dealerova viditelná karta
                    - "cards" (list[int]): Karty hráče
                    - "total" (int): Součet karet hráče
                    - "pair" (bool): True, pokud má hráč pár
                    - "soft" (bool): True, pokud má hráč měkkou ruku (s esem jako 11)
                    - "insurance_flag" (int): 1 nebo 0, zda má hráč pojištění
                    - "hilo_idx" (int): Hi-Lo index
                    - "init_deal" (int): 1, pokud se jedná o první rozdání
                    - "cards_dealt" (int): Počet již rozdaných karet
                    - "cards_seen_cnt" (dict): Počet výskytů jednotlivých karet

        Returns:
            str: Doporučená akce hráče podle základní strategie.
                Jedna z: "HIT", "STAND", "DOUBLE", "SPLIT", nebo None.
        """
        print(i_round_data)
        move = None
        dealer = i_round_data["d_show"]
        card = i_round_data["cards"][1]
        total = i_round_data["total"]
        # SPLIT: mám pár -> ['SPLIT'][hráčová karty][dealerová karta]
        if i_round_data["pair"]:
            print("pair")
            split_strategy = self.strategy["SPLIT"]
            if split_strategy.get(card):
                if split_strategy[card].get(dealer):
                    move = split_strategy[card].get(dealer)
        if not move:
            print("move")
            if i_round_data["soft"]:
                print("soft")
                double_s_strategy = self.strategy["DOUBLE_S"]
                hit_s_strategy = self.strategy["HIT_S"]
                # DOUBLE SOFT: mám soft, ['DOUBLE_S'][hráčův total][dealerová karta]
                if double_s_strategy.get(total):
                    if double_s_strategy[total].get(dealer):
                        move = double_s_strategy[total].get(dealer)
                # HIT SOFT: mám soft, ['HIT_S'][hráčův total][dealerová karta]
                if not move and hit_s_strategy.get(total):
                    if hit_s_strategy[total].get(dealer):
                        move = hit_s_strategy[total].get(dealer)
            # DOUBLE : ['DOUBLE'][hráčův total][dealerová karta]
            double_strategy = self.strategy["DOUBLE"]
            if not move and double_strategy.get(total):
                if double_strategy[total].get(dealer):
                    move = double_strategy[total].get(dealer)
            # HIT : ['HIT'][hráčův total][dealerová karta]
            hit_strategy = self.strategy["HIT"]
            if not move and hit_strategy.get(total):
                if hit_strategy[total].get(dealer):
                    move = hit_strategy[total].get(dealer)
        return move