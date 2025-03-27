# fenics/attack/attack_factory.py

from typing import Dict, Type, Optional, Any
import logging

from fenics.attack.attack_types.base import Attack
from fenics.attack.attack_types.poison import PoisonAttack
from fenics.attack.attack_types.delay import DelayAttack


class AttackFactory:
    """
    Factory class for creating attack instances.
    This makes it easy for users to select different attack types.
    """
    
    # Registry of available attacks
    _attacks: Dict[str, Type[Attack]] = {
        'poison': PoisonAttack,
        'delay': DelayAttack,
    }
    
    @classmethod
    def register_attack(cls, name: str, attack_class: Type[Attack]) -> None:
        """
        Register a new attack type.
        
        Args:
            name: Name of the attack
            attack_class: Attack class
        """
        cls._attacks[name] = attack_class
    
    @classmethod
    def get_attack(cls, attack_name: str, node_id: int, logger: Optional[logging.Logger] = None, **kwargs) -> Attack:
        """
        Get an attack instance by name.
        
        Args:
            attack_name: Name of the attack
            node_id: ID of the attacker node
            logger: Logger instance
            **kwargs: Additional arguments to pass to the attack constructor
            
        Returns:
            Instance of the requested attack
            
        Raises:
            ValueError: If the attack name is not recognized
        """
        if attack_name not in cls._attacks:
            available_attacks = list(cls._attacks.keys())
            raise ValueError(f"Unknown attack: '{attack_name}'. Available attacks: {available_attacks}")
        
        attack_class = cls._attacks[attack_name]
        return attack_class(node_id=node_id, logger=logger, **kwargs)
    
    @classmethod
    def list_available_attacks(cls) -> Dict[str, Type[Attack]]:
        """
        List all available attacks.
        
        Returns:
            Dictionary mapping attack names to attack classes
        """
        return cls._attacks.copy()