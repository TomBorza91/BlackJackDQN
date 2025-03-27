import player_BS

class PlayerCC:
    """
    Třída pro hráče využívajícího strategii počítání karet (Card Counting – CC).
    
    Strategie využívá Hi-Lo index ke zlepšení rozhodování oproti základní strategii.
    Pokud není nalezena konkrétní akce pro danou situaci nebo není jistota, použije se základní strategie (BS).
    """
    def __init__(self):
        """
        Inicializuje hráče s počítací strategií a instancí základní strategie pro zálohu.
        """
        self.p_BS = player_BS.PlayerBS()
        self.strategy = self.get_strategy()


    def get_strategy(self):
        """
        Vrací tabulku rozhodovacích Hi-Lo prahů pro jednotlivé situace ve hře.

        Returns:
            dict: Vnořený slovník, kde klíče představují typ akce (např. 'HIT', 'DOUBLE'),
                  a hodnoty obsahují Hi-Lo prahy pro jednotlivé kombinace hráč–dealer.
                  Pokud je hodnota 9999, akce se neprovádí. 'TODO' znamená výjimku.
        """
        strategy_data = {
            "HIT": {
                12: {
                    11: 9999,
                    2: 14,
                    3: 6,
                    4: 2,
                    5: -1,
                    6: 0,
                    7: 9999,
                    8: 9999,
                    9: 9999,
                    10: 9999,
                },
                13: {
                    11: 9999,
                    2: 1,
                    3: -2,
                    4: -5,
                    5: -9,
                    6: -8,
                    7: 50,
                    8: 9999,
                    9: 9999,
                    10: 9999,
                },
                14: {
                    11: 9999,
                    2: -5,
                    3: -8,
                    4: -13,
                    5: -17,
                    6: -17,
                    7: 20,
                    8: 38,
                    9: 9999,
                    10: 9999,
                },
                15: {
                    11: 16,
                    2: -12,
                    3: -17,
                    4: -21,
                    5: -26,
                    6: -28,
                    7: 13,
                    8: 15,
                    9: 12,
                    10: 8,
                },
                16: {
                    11: 14,
                    2: -21,
                    3: -25,
                    4: -30,
                    5: -34,
                    6: -35,
                    7: 10,
                    8: 11,
                    9: 6,
                    10: 0,
                },
                17: {
                    11: -15,
                    2: -9999,
                    3: -9999,
                    4: -9999,
                    5: -9999,
                    6: -9999,
                    7: -9999,
                    8: -9999,
                    9: -9999,
                    10: -9999,
                },
                18: {
                    11: -9999,
                    2: -9999,
                    3: -9999,
                    4: -9999,
                    5: -9999,
                    6: -9999,
                    7: -9999,
                    8: -9999,
                    9: -9999,
                    10: -9999,
                },
                19: {
                    11: -9999,
                    2: -9999,
                    3: -9999,
                    4: -9999,
                    5: -9999,
                    6: -9999,
                    7: -9999,
                    8: -9999,
                    9: -9999,
                    10: -9999,
                },
                20: {
                    11: -9999,
                    2: -9999,
                    3: -9999,
                    4: -9999,
                    5: -9999,
                    6: -9999,
                    7: -9999,
                    8: -9999,
                    9: -9999,
                    10: -9999,
                },
                21: {
                    11: -9999,
                    2: -9999,
                    3: -9999,
                    4: -9999,
                    5: -9999,
                    6: -9999,
                    7: -9999,
                    8: -9999,
                    9: -9999,
                    10: -9999,
                },
            },
            "HIT_S": {
                17: {
                    11: 9999,
                    2: 9999,
                    3: 9999,
                    4: 9999,
                    5: 9999,
                    6: 9999,
                    7: 29,
                    8: 9999,
                    9: 9999,
                    10: 9999,
                },
                18: {
                    11: -6,
                    2: -9999,
                    3: -9999,
                    4: -9999,
                    5: -9999,
                    6: -9999,
                    7: -9999,
                    8: -9999,
                    9: 9999,
                    10: 12,
                },
                19: {
                    11: -9999,
                    2: -9999,
                    3: -9999,
                    4: -9999,
                    5: -9999,
                    6: -9999,
                    7: -9999,
                    8: -9999,
                    9: -9999,
                    10: -9999,
                },
                20: {
                    11: -9999,
                    2: -9999,
                    3: -9999,
                    4: -9999,
                    5: -9999,
                    6: -9999,
                    7: -9999,
                    8: -9999,
                    9: -9999,
                    10: -9999,
                },
                21: {
                    11: -9999,
                    2: -9999,
                    3: -9999,
                    4: -9999,
                    5: -9999,
                    6: -9999,
                    7: -9999,
                    8: -9999,
                    9: -9999,
                    10: -9999,
                },
            },
            "DOUBLE": {
                5: {
                    11: 9999,
                    2: 9999,
                    3: 9999,
                    4: 9999,
                    5: 20,
                    6: 26,
                    7: 9999,
                    8: 9999,
                    9: 9999,
                    10: 9999,
                },
                6: {
                    11: 9999,
                    2: 9999,
                    3: 9999,
                    4: 27,
                    5: 18,
                    6: 24,
                    7: 9999,
                    8: 9999,
                    9: 9999,
                    10: 9999,
                },
                7: {
                    11: 9999,
                    2: 9999,
                    3: 45,
                    4: 21,
                    5: 14,
                    6: 17,
                    7: 9999,
                    8: 9999,
                    9: 9999,
                    10: 9999,
                },
                8: {
                    11: 9999,
                    2: 9999,
                    3: 22,
                    4: 11,
                    5: 5,
                    6: 5,
                    7: 22,
                    8: 9999,
                    9: 9999,
                    10: 9999,
                },
                9: {
                    11: 9999,
                    2: 3,
                    3: 0,
                    4: -5,
                    5: -10,
                    6: -12,
                    7: 4,
                    8: 14,
                    9: 9999,
                    10: 9999,
                },
                10: {
                    11: 6,
                    2: -15,
                    3: -17,
                    4: -21,
                    5: -24,
                    6: -26,
                    7: -17,
                    8: -9,
                    9: -3,
                    10: 7,
                },
                11: {
                    11: -3,
                    2: -23,
                    3: -26,
                    4: -29,
                    5: -33,
                    6: -35,
                    7: -26,
                    8: -16,
                    9: -10,
                    10: -9,
                },
            },
            "DOUBLE_S": {
                13: {2: 9999, 3: 10, 4: 2, 5: -19, 6: -13},
                14: {2: 9999, 3: 11, 4: -3, 5: -13, 6: -19},
                15: {2: 9999, 3: 19, 4: -7, 5: -16, 6: -23},
                16: {2: 9999, 3: 21, 4: -6, 5: -16, 6: -32},
                17: {2: "TODO", 3: -6, 4: -14, 5: -28, 6: -30},
                18: {2: 9999, 3: -2, 4: -15, 5: -18, 6: -23},
                19: {2: 9999, 3: 9, 4: 5, 5: 1, 6: 0},
                20: {2: 9999, 3: 20, 4: 12, 5: 8, 6: 8},
            },
            "SPLIT": {
                1: {
                    11: -17,
                    2: -9999,
                    3: -9999,
                    4: -9999,
                    5: -9999,
                    6: -9999,
                    7: -33,
                    8: -24,
                    9: -22,
                    10: -20,
                },
                2: {
                    11: 9999,
                    2: -9,
                    3: -15,
                    4: -22,
                    5: -30,
                    6: -9999,
                    7: -9999,
                    8: 9999,
                    9: 9999,
                    10: 9999,
                },
                3: {
                    11: 9999,
                    2: -21,
                    3: -34,
                    4: -9999,
                    5: -9999,
                    6: -9999,
                    7: -9999,
                    8: "TODO",
                    9: 9999,
                    10: 9999,
                },
                4: {
                    11: 9999,
                    2: 9999,
                    3: 18,
                    4: 8,
                    5: 0,
                    6: 9999,
                    7: 9999,
                    8: 9999,
                    9: 9999,
                    10: 9999,
                },
                5: {
                    11: 9999,
                    2: 9999,
                    3: 9999,
                    4: 9999,
                    5: 9999,
                    6: 9999,
                    7: 9999,
                    8: 9999,
                    9: 9999,
                    10: 9999,
                },
                6: {
                    11: 9999,
                    2: 0,
                    3: -3,
                    4: -8,
                    5: -13,
                    6: -16,
                    7: -8,
                    8: 9999,
                    9: 9999,
                    10: 9999,
                },
                7: {
                    11: 9999,
                    2: -22,
                    3: -29,
                    4: -35,
                    5: -9999,
                    6: -9999,
                    7: -9999,
                    8: -9999,
                    9: 9999,
                    10: 9999,
                },
                8: {
                    11: -18,
                    2: -9999,
                    3: -9999,
                    4: -9999,
                    5: -9999,
                    6: -9999,
                    7: -9999,
                    8: -9999,
                    9: -9999,
                    10: "TODO",
                },
                9: {
                    11: 10,
                    2: -3,
                    3: -8,
                    4: -10,
                    5: -15,
                    6: -14,
                    7: 8,
                    8: -16,
                    9: -22,
                    10: 9999,
                },
                10: {
                    11: 9999,
                    2: 25,
                    3: 17,
                    4: 10,
                    5: 6,
                    6: 7,
                    7: 19,
                    8: 9999,
                    9: 9999,
                    10: 9999,
                },
            },
        }
        return strategy_data
    
    
    def get_bet(self, i_idx, i_min=1, i_max=10):
        """
        Rozhodne výši sázky na základě aktuálního Hi-Lo indexu.
        Používá konzervativní strategii sázení, aby nedošlo k podezření.

        Args:
            i_idx (int): Aktuální Hi-Lo index.
            i_min (int): Minimální povolená sázka.
            i_max (int): Maximální povolená sázka (nepoužívá se přímo).

        Returns:
            float: Výše sázky jako násobek i_min (např. 1.5 * i_min).
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
        Vrací akci hráče na základě aktuální situace a Hi-Lo indexu.

        Args:
            i_round_data (dict): Slovník popisující stav hry.
                Obsahuje:
                    - "d_show" (int): Dealerova viditelná karta
                    - "cards" (list[int]): Hráčovy karty
                    - "total" (int): Součet hráčovy ruky
                    - "pair" (bool): True, pokud má hráč pár
                    - "soft" (bool): True, pokud ruka obsahuje eso započítané jako 11
                    - "insurance_flag" (int): Indikace pojištění (0/1)
                    - "hilo_idx" (int): Aktuální Hi-Lo index
                    - "init_deal" (int): 1 pokud jde o první rozdání
                    - "cards_dealt" (int): Počet rozdaných karet
                    - "cards_seen_cnt" (dict): Mapa výskytu jednotlivých hodnot

        Returns:
            str: Rozhodnutá akce hráče.
                Možné hodnoty: "HIT", "STAND", "DOUBLE", "SPLIT", "INSURANCE"
        """
        move = None
        dealer = i_round_data["d_show"]
        card = i_round_data["cards"][1]
        total = i_round_data["total"]
        hilo_idx = i_round_data['hilo_idx']
        if i_round_data["insurance_flag"] == 0 and dealer == 11 and hilo_idx > 8:
            move = 'INSURANCE'
            print(move)
    
        # Mám pár, mám provést split?
        if not move and i_round_data["pair"]:
            split_strategy = self.strategy["SPLIT"]
            if split_strategy.get(card):
                if split_strategy[card].get(dealer) is not None:
                    idx = split_strategy[card].get(dealer)
                    if idx != 'TODO':
                        if hilo_idx > idx:
                            move = 'SPLIT'
                    else:
                        # vyjimky
                        if card == 8 and dealer == 10 and hilo_idx < 24:
                            move = 'SPLIT'
                        if card == 3 and dealer == 8 and (6 < hilo_idx or hilo_idx < -2):
                            move = 'SPLIT'
    
        if not move:
            # Mám soft, mám provést Double nebo HIT?
            if i_round_data["soft"]:
                double_s_strategy = self.strategy["DOUBLE_S"]
                hit_s_strategy = self.strategy["HIT_S"]
                if double_s_strategy.get(total):
                    if double_s_strategy[total].get(dealer) is not None:
                        idx = double_s_strategy[total].get(dealer)
                        if idx != 'TODO':
                            if hilo_idx > idx:
                                move = 'DOUBLE'
                            else:
                                # vyjimky
                                if total == 17 and dealer == 2 and hilo_idx > 2 and hilo_idx < 10:
                                    move = 'DOUBLE'
                        
                if not move and hit_s_strategy.get(total):
                    if hit_s_strategy[total].get(dealer) is not None:
                        idx = hit_s_strategy[total].get(dealer)
                        if hilo_idx > idx:
                            move = 'STAND'
                        else:
                            move = 'HIT'
            # mám provést Double?
            double_strategy = self.strategy["DOUBLE"]
            if not move and double_strategy.get(total):
                if double_strategy[total].get(dealer) is not None:
                    idx = double_strategy[total].get(dealer)
                    if hilo_idx > idx:
                        move = 'DOUBLE'
            # mám provést HIT??
            hit_strategy = self.strategy["HIT"]
            if not move and hit_strategy.get(total):
                if hit_strategy[total].get(dealer) is not None:
                    idx = hit_strategy[total].get(dealer)
                    if hilo_idx > idx:
                        move = 'STAND'
                    else:
                        move = 'HIT'
            # furt nic, tak proveď základní strategii a upozorni mě o tom do logu
            if not move:
                print('BS')
                f = open("log2.txt", "a")
                f.write('-----------------------------')
                f.write(str(i_round_data))
                f.write('BS')
                f.write('-----------------------------')
                f.close()
                move = self.p_BS.get_move(i_round_data)
        return move
    