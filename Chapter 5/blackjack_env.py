import random
from typing import Tuple, Optional

class BlackjackEnv:
    """
    Blackjack game environment implementing standard casino rules.
    
    Game Rules:
    - Player and dealer each start with 2 cards
    - Goal: achieve hand sum as close to 21 as possible without exceeding it
    - Card values: Ace=1/11 (whichever is better), Jack/Queen/King=10, numbered cards (2-10)=their number
    - Player acts first (hit/stick), dealer follows fixed strategy: hit if sum < 17, stick if sum ≥ 17
    - Cards drawn from infinite deck with replacement
    
    State Representation:
    - player_sum: current player hand total (≥2)
    - dealer_showing: dealer's visible card (1-10)
    - usable_ace: whether player has ace counting as 11
    
    Actions:
    - 0: stick (stop taking cards)
    - 1: hit (take another card)
    
    Rewards:
    - +1: player wins
    - -1: player loses
    - 0: draw/tie
    
    Features:
    - Standard random initialization
    - Custom state initialization support
    - Proper ace handling for soft/hard hands
    """
    
    # Constants
    BUST_THRESHOLD = 21
    ACE_HIGH = 11
    ACE_LOW = 1
    DEALER_THRESHOLD = 17
    FACE_VALUE = 10
    
    # Actions
    STICK = 0
    HIT = 1
    
    # Rewards
    WIN = 1
    LOSE = -1
    DRAW = 0
    CONTINUE = 0
    
    def __init__(self):
        self.reset()
    
    def _validate_state(self, state: Tuple[int, int, bool]) -> None:
        """
        Validate state parameters for proper game conditions
        
        Ensures that the provided state tuple contains valid values that can occur
        in a real Blackjack game. This includes checking tuple structure, value ranges,
        and logical consistency between player sum and ace usage.
        
        Args:
            state: Tuple (player_sum, dealer_showing, usable_ace) representing game state
            
        Raises:
            ValueError: If tuple structure is invalid
            ValueError: If dealer_showing is outside valid range (1-10)
            ValueError: If player_sum is incompatible with usable_ace flag
        """

        if len(state) != 3:
            raise ValueError(f"state must be a tuple of 3 elements, got {len(state)}")
        
        player_sum, dealer_showing, usable_ace = state
        
        # Validate dealer_showing
        if not (1 <= dealer_showing <= 10):
            raise ValueError(f"dealer_showing must be 1-10, got {dealer_showing}")
        
        # Validate player_sum based on usable_ace
        if usable_ace:
            # With usable ace (ace=11), minimum is 13 (ace+2), maximum is 21 (ace+10)
            if not (13 <= player_sum <= 21):
                raise ValueError(f"player_sum with usable_ace must be 13-21, got {player_sum}")
        else:
            # Without usable ace, minimum is 4 (2+2), maximum is 20 (10+10)
            if not (4 <= player_sum <= 20):
                raise ValueError(f"player_sum without usable_ace must be 4-20, got {player_sum}")

    def reset(self, initial_state: Optional[Tuple[int, int, bool]] = None) -> Tuple[int, int, bool]:
        """
        Reset environment and return initial state
        
        Args:
            initial_state: Optional tuple (player_sum, dealer_showing, usable_ace)
                        for custom initialization. If None, uses random initialization.
        
        Returns:
            Tuple of (player_sum, dealer_showing, usable_ace)
        """

        if initial_state is None:
            # Random initialization
            self.player_cards = [self._draw_card(), self._draw_card()]
            self.dealer_cards = [self._draw_card(), self._draw_card()]
            
            self.player_sum, self.usable_ace = self._get_hand_value(self.player_cards)
            self.dealer_showing = self.dealer_cards[0]
        else:
            # Custom initialization with specific state
            self._validate_state(initial_state)
            
            player_sum, dealer_showing, usable_ace = initial_state
            
            self.player_sum = player_sum
            self.dealer_showing = dealer_showing
            self.usable_ace = usable_ace
            
            # Set up dealer's first card
            self.dealer_cards = [dealer_showing]
            
            # Construct player cards that match the desired state
            self.player_cards = self._construct_player_cards(player_sum, usable_ace)
        
        return (self.player_sum, self.dealer_showing, self.usable_ace)
    
    def step(self, action: int) -> Tuple[Tuple[int, int, bool], float, bool]:
        """
        Execute player action and return game response
        
        Args:
            action: Player action (0=STICK, 1=HIT)
            
        Returns:
            Tuple of (state, reward, done) where:
            - state: (player_sum, dealer_showing, usable_ace)
            - reward: game reward (WIN/LOSE/DRAW/CONTINUE)
            - done: whether episode has ended
            
        Raises:
            ValueError: If action is not 0 or 1
        """

        if action not in [self.STICK, self.HIT]:
            raise ValueError(f"Invalid action: {action}. Must be {self.STICK} (stick) or {self.HIT} (hit)")
        
        current_state = (self.player_sum, self.dealer_showing, self.usable_ace)
        
        if action == self.HIT:
            # Player takes another card
            self.player_cards.append(self._draw_card())
            self.player_sum, self.usable_ace = self._get_hand_value(self.player_cards)
            
            new_state = (self.player_sum, self.dealer_showing, self.usable_ace)
            
            if self.player_sum > self.BUST_THRESHOLD:  # Player busts
                return new_state, self.LOSE, True
            else:
                return new_state, self.CONTINUE, False
        
        else:  # action == self.STICK
            # Player stops, dealer plays
            reward = self._dealer_play()
            return current_state, reward, True
    
    def _draw_card(self) -> int:
        """
        Draw card from infinite deck with replacement
        
        Returns card value where:
        - Ace = 1
        - Number cards (2-9) = face value  
        - Ten/Jack/Queen/King = 10
        """
        # Use min to cap face cards (J=11, Q=12, K=13) at 10
        return min(self.FACE_VALUE, random.randint(1, 13))
    
    def _get_hand_value(self, cards: list) -> Tuple[int, bool]:
        """
        Calculate total hand value and determine ace usage
        
        Computes the optimal hand value by treating aces as 1 or 11 (whichever is better).
        An ace is considered "usable" when it's counted as 11 without causing a bust.
        
        Args:
            cards: List of card values
            
        Returns:
            Tuple of (hand_total, usable_ace_flag)
        """
        total = sum(cards)
        aces = cards.count(self.ACE_LOW)
        usable_ace = False
        
        # Use ace as 11 if beneficial and doesn't cause bust
        if aces > 0 and total + (self.ACE_HIGH - self.ACE_LOW) <= self.BUST_THRESHOLD:
            total += (self.ACE_HIGH - self.ACE_LOW)
            usable_ace = True
        
        return total, usable_ace
    
    def _construct_player_cards(self, target_sum: int, usable_ace: bool) -> list:
        """
        Construct exactly 2 player cards that result in target_sum with specified ace usage
        
        Args:
            target_sum: Desired player sum
            usable_ace: Whether ace should be counted as 11
            
        Returns:
            List of exactly 2 cards that produce the target sum and ace usage
        """
        if usable_ace:
            second_card = target_sum - self.ACE_HIGH  # target_sum - 11
            return [1, second_card]  # [Ace, second_card]

        else:
            # No usable ace - must construct without any ace
            # Valid range: 4-20 (2+2 to 10+10)
            

            for card1 in range(2, 11):  
                card2 = target_sum - card1
                if 2 <= card2 <= 10:
                    return [card1, card2]

    def _construct_player_cards(self, target_sum: int, usable_ace: bool, randomize: bool = False) -> list:
        """
        Creates a valid 2-card starting hand matching the desired sum and ace state.
        For usable ace hands, constructs ace + other card. For non-usable ace hands,
        constructs two non-ace cards with randomized selection or first-match option.
        
        Args:
            target_sum: Desired player hand total
            usable_ace: Whether the hand should contain a usable ace (ace=11)
            randomize: Whether to randomly select a valid non-ace pair (default True)
            
        Returns:
            List of exactly 2 card values that produce target_sum with correct ace usage
        """
        if usable_ace:
            # Hand contains ace counted as 11
            second_card = target_sum - self.ACE_HIGH
            return [self.ACE_LOW, second_card]

        else:
            # With only 2 cards, if we include an ace, it will always be usable
            # So we must avoid ace entirely          
            
            valid_pairs = []
            for card1 in range(2, 11):
                card2 = target_sum - card1
                if 2 <= card2 <= 10:
                    if not randomize:
                        return [card1, card2]  # Return first valid match immediately
                    valid_pairs.append([card1, card2])
            
            return random.choice(valid_pairs)

        
    def _dealer_play(self) -> float:
        """
        Execute dealer's turn following standard casino rules
        
        Dealer follows fixed strategy: hit if sum < 17, stick if sum >= 17.
        After dealer completes their hand, determines game outcome by comparing
        final hand values.
        
        Returns:
            Game reward from player's perspective (WIN/LOSE/DRAW)
        """

        dealer_sum, _ = self._get_hand_value(self.dealer_cards)
        
        # Dealer hits on < 17, sticks on >= 17
        while dealer_sum < self.DEALER_THRESHOLD:
            self.dealer_cards.append(self._draw_card())
            dealer_sum, _ = self._get_hand_value(self.dealer_cards)
        
        # Determine winner
        if dealer_sum > self.BUST_THRESHOLD:  # Dealer busts
            return self.WIN
        elif self.player_sum > dealer_sum:
            return self.WIN
        elif self.player_sum < dealer_sum:
            return self.LOSE
        else:
            return self.DRAW