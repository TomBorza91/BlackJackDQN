from os import truncate
import random
import math
from collections import Counter
import matplotlib.pyplot as plt
import pickle

"""
Simulace BlackJacku s podporou počítání karet

Tento modul implementuje základní simulaci karetní hry BlackJack, včetně mechaniky
počítání karet metodou Hi-Lo. Podporuje strategické rozhodování hráčů, sázení, pojištění,
splitování a další klíčové aspekty hry.

Třídy:
------
- Deck: Reprezentuje balíček (nebo více balíčků) karet pro BlackJack.
  Sleduje vytažené karty, počítá Hi-Lo index, umožňuje míchání (náhodné i deterministické).

- Hand: Reprezentuje ruku hráče nebo dealera. Umožňuje přidávání karet, výpočet hodnoty,
  určení bustu, BlackJacku, měkké ruky (soft hand) a možnosti splitu.

- Player: Reprezentuje hráče se zůstatkem na účtu. Rozhoduje se na základě zvolené strategie,
  podporuje různé herní akce. Sleduje statistiky a historii tahů.

- Dealer: Reprezentuje dealera. Hraje podle pevných pravidel (bere karty, dokud nemá aspoň 17).

- BlackjackGame: Řídí celou hru – rozdávání karet, zpracování rozhodnutí hráčů, vyhodnocení výsledků,
  aktualizaci zůstatků, sběr statistik a uložení dat o hře.

Implementované hráčské akce:
----------------------------
- **HIT**: Hráč si vezme další kartu.
- **STAND**: Hráč zůstane stát a ukončí tah.
- **DOUBLE**: Hráč zdvojnásobí sázku, vezme si ještě jednu kartu a končí tah.
- **SPLIT**: Pokud má hráč dvě karty stejné hodnoty, může je rozdělit do dvou samostatných ruk.
- **INSURANCE**: Pokud dealerova první karta je eso, může hráč uzavřít pojištění proti BlackJacku dealera.

Funkce:
-------
- Simulace celé hry BlackJack s více hráči.
- Výpočet Hi-Lo indexu včetně práce se skrytou kartou dealera.
- Rozhodování hráčů podle strategie (vstupní objekt).
- Evidence sázek, výher, proher, remíz a pojištění.
- Ukládání průběhu a výsledků hry pomocí `pickle`.

Použití:
--------
Vytvoř instanci třídy `BlackjackGame`, nastav hráče a počet balíčků, a zavolej metodu `play(rounds)`
pro simulaci zadaného počtu kol.

Příklad:
--------
game = BlackjackGame(num_decks=6, players=seznam_hracu, starting_balance=10000)  
vysledky = game.play(rounds=1000000)

Závislosti:
-----------
- random
- math
- collections.Counter
- matplotlib.pyplot
- pickle
"""

class Deck:
    """
    Reprezentuje balíček karet pro hru BlackJack, s podporou více balíčků a počítáním Hi-Lo.
    """
    def __init__(self, num_decks, random_f=True):
        """
        Inicializuje balíček s možností náhodného nebo deterministického míchání.

        Args:
            num_decks (int): Počet balíčků v hře.
            random_f (bool): True pro náhodné míchání, False pro deterministické (se seedem).
        """
        self.random_f = random_f
        self.num_decks = num_decks
        self.hilo = 0
        self.hilo_f = 0
        self.hilo_idx = 0
        self.cards = [2,3,4,5,6,7,8,9,10,10,10,10, 11] * 4 * num_decks
        self.hidden = None
        self.card_seen_cnt = {2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0,10:0,11:0}
        self.seed = None
        if not self.random_f:
            self.seed = 1000000
            random.seed(self.seed)
        self.shuffle()

    def add_seen_card(self, card):
        """
        Zaznamená kartu jako již vytaženou z balíčku.

        Args:
            card (int): Hodnota karty (2–11).
        """
        self.card_seen_cnt[card] += 1

    def refresh_hidden(self):
        """Odkryje dealerovu skrytou kartu a započítá ji do statistik."""
        self.add_seen_card(self.hidden)
        self.hidden = None

    def __eval_hilo(self, card):
        """
        Vrátí Hi-Lo hodnotu karty.

        Args:
            card (int): Hodnota karty.

        Returns:
            int: +1 pro nízké karty, 0 pro střední, -1 pro vysoké.
        """
        if card in [2, 3, 4, 5, 6]:
            return 1
        elif card in [7, 8, 9]:
            return 0
        else :
            return -1

    def __calc_hilo(self, card):
        """
        Aktualizuje Hi-Lo skóre a přepočítá index podle počtu zbývajících karet.

        Args:
            card (int): Nově rozdaná karta.
        """
        self.hilo += self.__eval_hilo(card)
        unseen_fix = 0
        unseen = 0
        if self.hidden is not None:
          unseen_fix = self.__eval_hilo(self.hidden)
          unseen = 1
        self.hilo_f = self.hilo - unseen_fix
        self.hilo_idx = math.floor((self.hilo_f / (len(self.cards) + unseen)) * 100)

    def get_hilo_idx(self):
        """
        Vrací normalizovaný Hi-Lo index.

        Returns:
            int: Hodnota Hi-Lo indexu škálovaná podle počtu zbývajících karet.
        """
        return self.hilo_idx

    def deal_card(self, hide=False):
        """
        Vrátí kartu z balíčku a aktualizuje počítání.

        Args:
            hide (bool): Pokud True, karta se uloží jako skrytá (pro dealera).

        Returns:
            int: Hodnota rozdané karty.
        """
        card = self.cards.pop()
        if hide:
            self.hidden = card
        else:
            self.add_seen_card(card)
        self.__calc_hilo(card)
        return card

    def count_dealt_cards(self):
        """
        Vrací počet již rozdaných karet.

        Returns:
            int: Počet rozdaných karet.
        """
        return (52*self.num_decks) - len(self.cards)

    def refresh_deck(self):
        """Obnoví balíček do původního stavu a promíchá ho."""
        self.cards = [2,3,4,5,6,7,8,9,10,10,10,10, 11] * 4 * self.num_decks
        self.card_seen_cnt = {2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0,10:0,11:0}
        self.hilo = 0
        self.hilo_f = 0
        self.hilo_idx = 0
        self.shuffle()

    def shuffle(self):
        """Promíchá balíček. U deterministického režimu zvyšuje seed."""
        random.shuffle(self.cards)
        if not self.random_f:
            self.seed += 1
            random.seed(self.seed)


class Hand:
    """
    Reprezentuje ruku hráče nebo dealera ve hře BlackJack.
    """
    def __init__(self):
        """Inicializuje prázdnou ruku se sázkou a pojištěním."""
        self.cards = []
        self.soft = 0
        self.bet = 0
        self.insurance_bet = 0

    def add_card(self, card):
        """
        Přidá kartu do ruky.

        Args:
            card (int): Hodnota karty.
        """
        self.cards.append(card)

    def get_cards(self):
        """
        Vrací karty v ruce.

        Returns:
            list[int]: Seznam karet.
        """
        return self.cards

    def value(self):
        """
        Spočítá hodnotu ruky s ohledem na esa.

        Returns:
            int: Celková hodnota ruky.
        """
        value = sum(self.cards)
        num_aces = self.cards.count(11)
        while value > 21 and num_aces:
            value -= 10
            num_aces -= 1
 
        if num_aces:
            self.soft = 1

        return value
    
    def make_bet(self, bet):
        """
        Nastaví (nebo přičte) sázku k této ruce.

        Args:
            bet (float): Výše sázky.
        """
        self.bet += bet

    def set_insurance(self, bet):
        """
        Nastaví sázku na pojištění.

        Args:
            bet (float): Výše pojištění.
        """
        self.insurance_bet += bet
 
    def insurance_flag(self):
        """
        Vrací, zda bylo uzavřeno pojištění.

        Returns:
            int: 1 pokud ano, 0 pokud ne.
        """
        return 1 if self.insurance_bet>0 else 0

    def is_bust(self):
        """
        Určí, zda ruka přetáhla (přes 21).

        Returns:
            bool: True pokud bust, jinak False.
        """
        return self.value() > 21
 
    def is_soft(self):
        """
        Určuje, zda je ruka měkká (obsahuje eso jako 11).

        Returns:
            int: 1 pokud ano, jinak 0.
        """
        return self.soft

    def can_split(self):
        """
        Určuje, zda lze ruku rozdělit.

        Returns:
            bool: True pokud ruka obsahuje dvě stejné karty.
        """
        return len(self.cards) == 2 and self.cards[0] == self.cards[1]

    def is_blackjack(self):
        """
        Určuje, zda je ruka BlackJack.

        Returns:
            bool: True pokud má ruka hodnotu 21 a obsahuje dvě karty.
        """
        return self.value() == 21 and len(self.cards) == 2
 
    def __repr__(self):
        """
        Vrací textovou reprezentaci ruky.

        Returns:
            str: Ruka a její hodnota jako text.
        """
        return f"{' '.join(map(str, self.cards))} (Value: {self.value()})"

class Player:
    """
    Reprezentuje hráče ve hře BlackJack, včetně zůstatku, statistik a rozhodovací strategie.
    """
    def __init__(self, name, strategy, balance):
        """
        Inicializuje hráče se jménem, strategií a počátečním zůstatkem.
        
        Args:
            name (str): Jméno hráče.
            strategy (object): Objekt strategie s metodami get_bet() a get_move().
            balance (float): Počáteční zůstatek.
        """
        self.name = name
        self.strategy = strategy
        self.hands = [Hand()]
        self.balance = balance
       # self.bet = 0
        self.lose_cnt = 0
        self.win_cnt = 0
        self.tie_cnt = 0
       # self.insurance_bet = 0
        self.bets = []
        self.loses = []
        self.wins = []
        self.move = {'HIT':self.hit,'STAND':self.stand, 'DOUBLE':self.double_down,'SPLIT':self.split, 'INSURANCE':self.insurance}
        self.move_memory = []

    def make_bet(self, i_idx):
        """
        Získá výši sázky ze strategie a provede sázku.

        Args:
            i_idx (int): Aktuální Hi-Lo index.
        """
        while True:
            try:
                ##self.bet = int(input(f"{self.name}, enter your bet (balance: {self.balance}): "))
                bet = self.strategy.get_bet(i_idx, 1, 10)
                self.hands[0].make_bet(bet)
                #print(self.bet)
                if 0 < bet <= self.balance:
                    self.balance -= bet
                    break
                else:
                    print("Invalid bet amount.")
            except ValueError:
                print("Invalid input. Please enter a number.")

    def hit(self, hand, deck):
        """
        Vezme kartu a rozhodne, zda může pokračovat.

        Args:
            hand (Hand): Aktuální ruka hráče.
            deck (Deck): Balíček karet.

        Returns:
            bool: False pokud hráč přetáhl, jinak True.
        """
        hand.add_card(deck.deal_card())
       # print(f"{self.name}'s hand: {hand}")
        if hand.is_bust():
        #    print(f"{self.name} busts!")
            return False
        return True

    def stand(self, hand, deck):
        """
        Hráč stojí, ukončuje tah.

        Returns:
            bool: False (vždy ukončuje tah).
        """
        return False

    def double_down(self, hand, deck):
        """
        Hráč zdvojnásobí sázku a vezme poslední kartu.

        Returns:
            bool: False (tah končí).
        """
        if self.balance >= hand.bet:
            self.balance -= hand.bet
            hand.bet *= 2
            hand.add_card(deck.deal_card())
          #  print(f"{self.name}'s hand: {hand}")
        return False

    def split(self, hand, deck):
        """
        Rozdělí ruku na dvě nové ruce.

        Returns:
            bool: True pokud split proběhl.
        """
        if hand.can_split() and self.balance >= hand.bet:
            self.balance -= hand.bet
            new_hand = Hand()
            new_hand.make_bet(hand.bet)
            new_hand.add_card(hand.cards.pop())
            new_hand.add_card(deck.deal_card())
            hand.add_card(deck.deal_card())
            self.hands.append(new_hand)
          #  print(f"{self.name} splits hand into {hand} and {new_hand}")
        else:
            print("Cannot split this hand.")
        return True

    def insurance(self, hand, deck):
        """
        Uzavře pojištění proti BlackJacku dealera.

        Args:
            hand (Hand): Aktuální ruka.
            deck (Deck): Balíček karet (není zde nutný, ale zůstává pro rozhraní).
        """
        insurance_amount = hand.bet / 2
        if self.balance >= insurance_amount:
            hand.insurance_bet = insurance_amount
            self.balance -= insurance_amount
        else:
            print(f"{self.name}, you do not have enough balance for insurance.")
    
    def make_decision(self, deck, visible_card):
        """
        V cyklu získává tahy od strategie a provádí je.

        Args:
            deck (Deck): Balíček karet.
            visible_card (int): Dealerova viditelná karta.
        """
        i = 0
        alive = True
        # round_data["total"], round_data["d_show"], round_data["cards_dealt"], round_data['hilo_idx'], round_data["pair"], init_deal
        while alive:
            hand = self.hands[i]
            init = True
            while True:
                init_deal = 1 if i==0 and init else 0
                #round_data = {"d_show":visible_card, "cards":hand.get_cards(), "total":hand.value(), "pair":hand.can_split(), "soft":hand.is_soft(), "insurance_flag":hand.insurance_flag(),"hilo_idx":deck.get_hilo_idx(),"init_deal":init_deal, "cards_dealt":deck.count_dealt_cards()}
                round_data = {
                    "d_show": visible_card,
                    "cards": hand.get_cards(),
                    "total": hand.value(),
                    "pair": hand.can_split() if len(self.hands)==1 else False,
                    "soft": hand.is_soft(),
                    "insurance_flag": hand.insurance_flag(),
                    "hilo_idx": deck.get_hilo_idx(),
                    "init_deal": 1 if i==0 and init else 0,
                    "cards_dealt": deck.count_dealt_cards(),
                    "cards_seen_cnt": deck.card_seen_cnt
                }
                action = self.strategy.get_move(round_data)
                if action in self.move:
                    self.move_memory.append(action)
                  #  print('ACTION: ',action)
                    if not self.move[action](hand, deck):
                        print('break')
                        break
                else:
                    print("Invalid action. Please choose again.")
                init=False
            i += 1
            if i >= len(self.hands):
                alive = False


class Dealer():
    """
    Reprezentuje dealera ve hře BlackJack.
    """
    def __init__(self):
        """Inicializuje dealera s prázdnou rukou."""
        self.name = "Dealer"
        self.hand = Hand()
        self.balance = float('inf')

    def play(self, deck):
        """
        Dealer odehraje svůj tah podle pravidel (bere do 17).

        Args:
            deck (Deck): Balíček karet.
        """        deck.refresh_hidden()
        while self.hand.value() < 17:
            self.hand.add_card(deck.deal_card())
      #  print(f"Dealer's hand: {self.hand}")

class BlackjackGame:
    """
    Třída pro řízení celé hry BlackJack – od rozdání po vyhodnocení.
    """
    def __init__(self, num_decks, players, starting_balance, random_f=True):
        """
        Inicializuje instanci hry.

        Args:
            num_decks (int): Počet balíčků v hře.
            players (list): Seznam hráčů ve formátu {'name': str, 'player': strategy_object}.
            starting_balance (float): Počáteční zůstatek každého hráče.
            random_f (bool): Pokud False, hra je deterministická (fixed seed).
        """
        num_players = len(players)
        self.deck = Deck(num_decks, random_f)
        self.dealer = Dealer()
        self.players = [Player(players[i]['name'], players[i]['player'], starting_balance) for i in range(num_players)]
        self.money = {players[i]['name']:[] for i in range(num_players)}
        self.game_memory = {'balance':None, 'lose_cnt':None, 'win_cnt':None, 'tie_cnt':None, 'bets':None, 'loses':None, 'wins':None, 'move_memory':None}

    def eval_hand(self, player, hand, dealer_value):
        """
        Vyhodnotí výsledek jedné ruky hráče vůči dealerovi.

        Args:
            player (Player): Hráč, jehož ruka se vyhodnocuje.
            hand (Hand): Jedna ruka hráče.
            dealer_value (int): Hodnota ruky dealera.
        """
        player_value = hand.value()
        player_blackjack = hand.is_blackjack()
        dealer_blackjack = dealer_value == 21 and len(self.dealer.hand.cards) == 2
        if dealer_blackjack:
          #  print("Dealer has Blackjack!")
            if hand.insurance_bet > 0:
                player.balance += hand.insurance_bet * 2
           #     print(f"{player.name} wins insurance bet of {hand.insurance_bet * 2}.")
        
        if player_value <= 21:
            if dealer_value > 21 or player_value > dealer_value:
                winnings = hand.bet * 2 if not player_blackjack else hand.bet * 2.5
                player.balance += winnings
                player.win_cnt += 1
              #  print(f"{player.name} wins {winnings} with hand {hand}!")
                player.wins.append(hand.bet if not player_blackjack else hand.bet * 1.5)
            elif player_value == dealer_value:
                if player_blackjack and not dealer_blackjack:
                    winnings = hand.bet * 2.5
                    player.balance += winnings
               #     print(f"{player.name} wins {winnings} with hand {hand}!")
                    player.win_cnt += 1
                    player.wins.append(hand.bet)
                elif not dealer_blackjack or (dealer_blackjack and player_blackjack):
                    player.balance += hand.bet                  
                #    print(f"{player.name} ties and gets back {hand.bet} with hand {hand}!")
                    player.tie_cnt += 1
                else:
                 #   print(f"{player.name} loses {hand.bet} with hand {hand}.")
                    player.lose_cnt += 1
                    player.loses.append(hand.bet)
            else:
                #print(f"{player.name} loses {hand.bet} with hand {hand}.")
                player.lose_cnt += 1
                player.loses.append(hand.bet)
        else:
            player.lose_cnt += 1
            player.loses.append(hand.bet)

    def play(self, rounds):
        """
        Spustí hru na zadaný počet kol.

        Args:
            rounds (int): Počet odehraných kol.

        Returns:
            dict: Historie zůstatků hráčů po každém kole.
        """
        round_count = 0

        while round_count <= rounds:
            round_count += 1
            print(f"--- Round {round_count} ---")
            for player in self.players:
                player.make_bet(self.deck.get_hilo_idx())

            self.dealer.hand.add_card(self.deck.deal_card())
            self.dealer.hand.add_card(self.deck.deal_card(hide=True))
            visible_card = self.dealer.hand.cards[0]
            print(f"Dealer shows: {visible_card}")

            for player in self.players:
                for hand in player.hands:
                    hand.add_card(self.deck.deal_card())
                    hand.add_card(self.deck.deal_card())
                print(f"{player.name}'s hand: {player.hands[0]}")
#                player.offer_insurance(self.deck, visible_card)
                player.make_decision(self.deck, visible_card)

            self.dealer.play(self.deck)
            dealer_value = self.dealer.hand.value()
            print(f"Dealer final hand: {self.dealer.hand}")

            for player in self.players:
                for hand in player.hands:
                    player.bets.append(hand.bet)
                    self.eval_hand(player, hand, dealer_value)
                    print('Player balance:', player.balance)
                    self.money[player.name].append(player.balance)
                    print('lose_cnt ' , player.lose_cnt)
                    print('win_cnt ' , player.win_cnt)
                    print('tie_cnt ' , player.tie_cnt)

                player.hands = [Hand()]

            self.dealer.hand = Hand()
            if len(self.deck.cards) < 52:
                self.deck.refresh_deck()
        for player in self.players:
            print('bets: ', sum(player.bets))
            print('loses: ', sum(player.loses))
            print('wins: ', sum(player.wins))
            self.game_memory['balance'] = player.balance
            self.game_memory['lose_cnt'] = player.lose_cnt
            self.game_memory['win_cnt'] = player.win_cnt
            self.game_memory['tie_cnt'] = player.tie_cnt
            self.game_memory['bets'] = player.bets
            self.game_memory['loses'] = sum(player.loses)
            self.game_memory['wins'] = sum(player.wins)
            self.game_memory['move_memory'] = player.move_memory
            self.game_memory['money'] = self.money
            with open('/content/drive/MyDrive/Diplomka/game_memory.pkl', 'wb') as f:
                pickle.dump(self.game_memory, f)
        return self.money