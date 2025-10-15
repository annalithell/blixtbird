# fenics/attack/attack_manager.py

import random
import logging
from typing import List, Dict, Set, Optional, Union, Tuple, Any
from fenics.attack.attack_factory import AttackFactory


class AttackManager:
    """
    A module to manage attacker nodes and their attack strategies in federated learning.
    """
    
    def __init__(self,
                 num_nodes: int,
                 use_attackers: bool = False,
                 all_nodes: Optional[Dict[int, str]] = None,
                 max_attacks: Optional[int] = None,
                 random_seed: int = 12345,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize the attack manager.
        
        Args:
            num_nodes: Total number of nodes in the network
            use_attackers: Flag to enable or disable attackers
            num_attackers: Number of attacker nodes to select (if attacker_nodes is None)
            attacker_nodes: List of specific node IDs to designate as attackers
            attacks: List of attack types to perform
            max_attacks: Maximum number of attacks per attacker
            random_seed: Random seed for reproducibility
            logger: Logger instance
        """
        self.num_nodes = num_nodes
        self.use_attackers = use_attackers
        self.all_nodes = all_nodes
        self.attacker_nodes = self.identify_attackers()
        self.max_attacks = max_attacks
        self.random_seed = random_seed
        self.logger = logger or logging.getLogger()
        
        # These will be set later
        self.attacker_participation_rounds = {}
        self.attacker_attack_rounds = {}
        
        # Initialize random number generator with fixed seed
        self.rng = random.Random(random_seed)
    
    def identify_attackers(self) -> Dict[int, str]:
        """
        Identify attacker nodes based on configuration.
        
        Returns:
            None
        """
        real_attackers = {}

        if not self.use_attackers:
            return real_attackers

        if self.all_nodes is not None:
            # TODO: Maybe would be better to init simulator list[nodes] here as well
            valid_attacks = AttackFactory.list_available_attacks()
            for id, node_type in self.all_nodes.items(): 
                if node_type in valid_attacks.values():
                    real_attackers[id]=node_type
                else:
                    self.logger.warning(f"Node {id} has unrecognized attack type '{node_type}' and will not be an attacker.")
                    
        self.logger.info(f"Attacker nodes: {self.attacker_nodes} with attacks: {self.attacker_nodes.values()}")
        return real_attackers

    def plan_attacks(self, participating_nodes_per_round: List[List[int]]) -> Dict[int, Set[int]]:
        """
        Plan when each attacker will perform attacks.
        
        Args:
            participating_nodes_per_round: List of lists of participating nodes for each round
            
        Returns:
            Dictionary mapping attacker node IDs to sets of rounds when they will attack
        """
        if not self.use_attackers or not self.attacker_nodes:
            return {}
        
        # Precompute the rounds in which each attacker participates
        self.attacker_participation_rounds = {attacker_id: [] for attacker_id in self.attacker_nodes.keys()}
        
        for rnd, participating_nodes in enumerate(participating_nodes_per_round, start=1):
            for attacker_id in self.attacker_nodes.keys():
                if attacker_id in participating_nodes:
                    self.attacker_participation_rounds[attacker_id].append(rnd)
        
        # For each attacker node, randomly select rounds to perform attacks
        self.attacker_attack_rounds = {}
        
        for attacker_id in self.attacker_nodes.keys():
            participation_rounds = self.attacker_participation_rounds[attacker_id]
            
            if self.max_attacks is None or self.max_attacks >= len(participation_rounds):
                # Attacker will perform attack in all participation rounds
                attack_rounds = participation_rounds
            else:
                # attack_rounds = self.rng.sample(participation_rounds, self.max_attacks)
                attack_rounds = random.sample(participation_rounds, self.max_attacks)
            
            self.attacker_attack_rounds[attacker_id] = set(attack_rounds)
            
            self.logger.info(f"Attacker {attacker_id} will attack in rounds: {sorted(attack_rounds)}")
        
        return self.attacker_attack_rounds
    
    def get_attack_type(self, node_id: int, round_num: int) -> Optional[str]:
        """
        Determine the attack type for a node in a specific round.
        
        Args:
            node_id: Node ID
            round_num: Round number
            
        Returns:
            Attack type or None if the node doesn't attack in this round
        """
        if not self.use_attackers:
            return None
            
        if round_num not in self.attacker_attack_rounds.get(node_id, set()):
            return None
            
        return self.attacker_nodes[node_id]
    
    def get_attacker_node_ids(self) -> List[int]:
        """ 
        Helper function to return the attacker nodes ids.
        Returns: 
            list[int]
        """
        return list(self.attacker_nodes.keys())
    
    def create_attack(self, node_id: int, attack_type: str):
        """
        Create an attack instance of the specified type.
        
        Args:
            node_id: Node ID
            attack_type: Type of attack to create
            
        Returns:
            Attack instance
        """
        return AttackFactory.get_attack(attack_type, node_id, self.logger)
    
    def is_attacker(self, node_id: int) -> bool:
        """
        Check if a node is an attacker.
        
        Args:
            node_id: Node ID
            
        Returns:
            True if the node is an attacker, False otherwise
        """
        return node_id in list(self.attacker_nodes.keys())