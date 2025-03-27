import random
import numpy as np
import copy
import pickle # ukládání a načítání souborů
import BlackJack
import DQN
from matplotlib import pyplot as plt
import time
import sys


class PlayerDQN(BlackJack.Player):
    """
    Hráč používající Double DQN agenta pro rozhodování v prostředí Blackjacku.

    Tento hráč sleduje pozorování prostředí, používá akční masky pro validní tahy
    a ukládá zkušenosti pro trénování modelu.
    """
    def __init__(self, name, strategy, balance):
        """
        Inicializuje instanci DQN hráče.

        Args:
            name (str): Jméno hráče.
            strategy: Není využito (z důvodu kompatibility s původním API).
            balance (float): Počáteční zůstatek hráče.
        """
        super().__init__(name, strategy, balance)
        self.agent = DQN.DQN(16, 5)
        print('123')
        self.agent.load_model()
        self.dqn_moves = {
            0:"HIT",
            1:"STAND",
            2:"DOUBLE",
            3:"SPLIT",
            4:"INSURANCE"
        }
        self.action_prev = None
        self.action = None
        self.observation_prev = None
        self.observation = None
        self.next_observation = None

    def make_bet(self, get_hilo_idx):
        self.hands[0].make_bet(1)
        if 0 < self.hands[0].bet <= self.balance:
            self.balance -= self.hands[0].bet
            
        else:
            print("Invalid bet amount.")
            
    def get_observation(self, round_data):
        """
        Převede herní stav na numerický vektor pro DQN.

        Args:
            round_data (dict): Slovník obsahující stav hry.

        Returns:
            list: Vektor pozorování jako vstup pro model.
        """
        observation = [
            round_data["total"], 
            round_data["d_show"], 
            round_data["cards_dealt"], 
            round_data['hilo_idx'], 
            round_data["cards_seen_cnt"][2], 
            round_data["cards_seen_cnt"][3], 
            round_data["cards_seen_cnt"][4], 
            round_data["cards_seen_cnt"][5], 
            round_data["cards_seen_cnt"][6], 
            round_data["cards_seen_cnt"][7], 
            round_data["cards_seen_cnt"][8], 
            round_data["cards_seen_cnt"][9], 
            round_data["cards_seen_cnt"][10], 
            round_data["cards_seen_cnt"][11], 
            round_data["pair"], 
            round_data["init_deal"]
        ]
        return observation

    def make_decision_dqn(self, deck, visible_card, r, round_rewards):
        """
        Hlavní rozhodovací smyčka hráče. Vykonává tahy podle výstupu DQN.

        Args:
            deck (Deck): Instance balíčku karet.
            visible_card (Card): Viditelná karta dealera.
            r (int): Počet aktuálního kola.
            round_rewards (list): Seznam odměn pro dané kolo.

        Returns:
            None
        """
        i = 0
        alive = True
        while alive:
            hand = self.hands[i]
            init = True
            if i == 1: #H2
                self.observation_prev = self.observation
                self.action_prev = self.action 
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
            if r == 1 or i == 1:
                self.observation = self.get_observation(round_data)
                self.observation = np.reshape(self.observation, [1,16])
            else:
                self.next_observation = self.get_observation(round_data)
                self.next_observation = np.reshape(self.next_observation, [1,16])
                if self.observation_prev is not None:
                    self.agent.save_memory(self.observation_prev, self.action_prev, round_rewards[0], self.next_observation, True)
                    self.agent.save_memory(self.observation, self.action, round_rewards[1], self.next_observation, True)
                else:
                    self.agent.save_memory(self.observation, self.action, round_rewards[0], self.next_observation, True)
                self.observation = self.next_observation
                self.observation_prev = None 
            while True:
                init_deal = 1 if i==0 and init else 0

                #MASK
                if round_data["total"] >= 20:
                    if init_deal:
                        if round_data["total"] == 21:
                            action_mask = np.array([0,1,0,0,0])
                        if round_data["total"] == 20:
                            action_mask = np.array([0,1,0,0,1])
                    else:
                        action_mask = np.array([0,1,0,0,0])
                else:
                    action_mask = np.array([1,1,1,1,1])
                if not round_data["pair"]:
                    action_mask[3] = 0
                if not init_deal or (init_deal and round_data["d_show"] != 11):
                    action_mask[4] = 0 

                self.action = self.agent.action(self.observation, action_mask, train=True)
                if (self.action == 3 and not round_data["pair"]) or (self.action == 4 and not init_deal):
                    print('FAUL!')
                    print(self.action)
                
                if self.dqn_moves[self.action] in self.move:
                    print('ACTION: ',self.action)
                    if not self.move[self.dqn_moves[self.action]](hand, deck):
                        print('break')
                        break

                    reward=0
                    round_data = {
                        "d_show": visible_card,
                        "cards": hand.get_cards(),
                        "total": hand.value(),
                        "pair": hand.can_split() if len(self.hands)==1 else False,
                        "soft": hand.is_soft(),
                        "insurance_flag": hand.insurance_flag(),
                        "hilo_idx": deck.get_hilo_idx(),
                        "init_deal": 0,
                        "cards_dealt": deck.count_dealt_cards(),
                        "cards_seen_cnt": deck.card_seen_cnt
                    } #"init_deal": 0 ptž next_observation už není init deal...
                    self.next_observation = self.get_observation(round_data)
                    self.next_observation = np.reshape(self.next_observation, [1,16])
                    self.agent.save_memory(self.observation, self.action,reward, self.next_observation, False)
                    #episode_reward += reward -- 0 neřeším
                    self.observation = self.next_observation
                else:
                    print("Invalid action. Please choose again.")
                init=False
            i += 1
            if i >= len(self.hands):
                alive = False 


class DQN_Trainer(BlackJack.BlackjackGame):
    """
    Třída umožňující trénink Double DQN agenta v prostředí hry Blackjack.

    Obsahuje loop pro odehrání her, uložení zkušeností, aktualizaci modelu a epsilon decay.

    """
    def __init__(self, rounds, num_decks, starting_balance):
        """
        Inicializuje prostředí hry s DQN hráčem.

        Args:
            rounds (int): Celkový počet kol.
            num_decks (int): Počet balíčků karet.
            starting_balance (float): Počáteční zůstatek hráče.
        """
        p = [{'name':'p_DQN', 'player':None}]
        super().__init__(num_decks, p, starting_balance)
        self.players = [PlayerDQN('p_DQN', None, starting_balance)]
        self.rewards = []
        self.train_counter = 0
        self.batch_size = 32
        self.target_update = 200
        self.reward = 0

    def reward_function(self, p_hand, d_hand, player_action, player):
        """
        Vypočítá odměnu pro danou kombinaci hráčovy a dealerovy ruky.

        Args:
            p_hand (Hand): Hráčova ruka.
            d_hand (Hand): Dealerova ruka.
            player_action (str): Poslední akce hráče.
            player (PlayerDQN): Instance hráče.

        Returns:
            float: Vypočtená odměna.
        """
        player_total = p_hand.value()
        dealer_total = d_hand.value()
        current_bet = p_hand.bet
        insurance_loss = 0
        original_bet = current_bet if player_action != 'DOUBLE' else current_bet / 2
        if p_hand.insurance_bet!=0:
            if d_hand.is_blackjack():
                insurance_loss = 2*original_bet
            else:
                insurance_loss = -2*original_bet #trest za ztrátu způsobenou pojištěním 
    
        if player_total > 21:
            return (-2 * current_bet) + insurance_loss
        elif (dealer_total > 21 or player_total > dealer_total) and not p_hand.is_blackjack():
            return (1 * current_bet) + insurance_loss # Výhra
        elif p_hand.is_blackjack() and not d_hand.is_blackjack(): # Výhra Blackjack
            return (1.5 * current_bet) + insurance_loss
        elif d_hand.is_blackjack() and not p_hand.is_blackjack(): # Prohra Blackjack
            return ((-1*current_bet) + insurance_loss)
        elif dealer_total > player_total: # Prohra
            return ((-1*current_bet) + insurance_loss)
        else:
            return 0 + insurance_loss # Remíza


    def train(self, rounds):
        """
        Hlavní trénovací smyčka pro DQN agenta.

        Args:
            rounds (int): Počet odehraných kol.

        Returns:
            dict: Historie financí hráče ve hře.
        """
        start_time = time.time()
        last_time = start_time
        round_count = 0
        round_rewards = []
        episode_reward = 0
        episode_earnings = 0
        observation = None
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
                player.make_decision_dqn(self.deck, visible_card, round_count, round_rewards)

            self.dealer.play(self.deck)
            dealer_value = self.dealer.hand.value()
            print(f"Dealer final hand: {self.dealer.hand}")

            player = self.players[0]
            h1 = player.hands[0]
            move_prev = player.dqn_moves[player.action_prev] if player.action_prev is not None else None
            move = player.dqn_moves[player.action]

            round_rewards = []
            if len(player.hands) == 1:
                round_rewards.append(self.reward_function(h1, self.dealer.hand, move, player))
            else:
                h2 = player.hands[1]
                reward2 = self.reward_function(h2, self.dealer.hand, move, player)
                round_rewards.append(self.reward_function(h1, self.dealer.hand, move, player))
                round_rewards.append(self.reward_function(h2, self.dealer.hand, move, player))

            for player in self.players:
                for hand in player.hands:
                    player.bets.append(hand.bet)
                    prev_ballance = player.balance
                    self.eval_hand(player, hand, dealer_value)
                    self.money[player.name].append(player.balance)
                    episode_earnings += player.balance - prev_ballance
                # Reset bets and hands for the next round
                #player.bet = 0
                #player.insurance_bet = 0
                player.hands = [BlackJack.Hand()]

            self.dealer.hand = BlackJack.Hand()
            if len(player.agent.memory) > self.batch_size:
#                curr_time2 = time.time()
                self.train_counter += 1 
                player.agent.train(self.batch_size)
 #               print("--- %s seconds ---" % (time.time() - curr_time2))
                if round_count % self.target_update == 0:
                    player.agent.copy_to_target()
            player.agent.epsilon_decay()
            self.rewards += round_rewards


            if round_count % 10 == 0:
                curr_time = time.time()
                print("episode: {}/{}".format(round_count, rounds))
                print("--- %s seconds ---" % (curr_time - last_time))
                last_time = curr_time
            # Check if the deck needs to be refreshed
            if len(self.deck.cards) < 52:
                self.deck.refresh_deck()
            if round_count % 30 == 0:
                player.agent.save_model()
                open_file = open('rewards.txt', "wb")
                pickle.dump(self.rewards, open_file)
                open_file.close()
            episode_reward = 0
            episode_earnings = 0

        player.agent.save_model()
        open_file = open('rewards.txt', "wb")
        pickle.dump(self.rewards, open_file)
        open_file.close()

        for player in self.players:
            print('Player balance:', player.balance)
            print('lose_cnt ' , player.lose_cnt)
            print('win_cnt ' , player.win_cnt)
            print('tie_cnt ' , player.tie_cnt)
            print('bets: ', sum(player.bets))
            print('loses: ', sum(player.loses))
            print('wins: ', sum(player.wins))
            print('train_counter:', self.train_counter)
            print('EPSILON:', player.agent.EPSILON)
        print("--- %s seconds ---" % (time.time() - start_time))
        return self.money

if __name__ == '__main__':
    dqn = DQN_Trainer(100000, 8, 500000)
    dqn.train(100000)


