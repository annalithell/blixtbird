# node/attackregistry.py
import importlib
import pkgutil
from pathlib import Path
from typing import Dict, Type
import os

ATTACK_REGISTRY: Dict[str, Type] = {}

def register_attack(class_name: str):

    """
    __USAGE__:
    @register_attack(<THE IMPLEMENTED CLASS NAME>)
    
    __DESCRIPTION__:
    decorator to register attack classes
    This has to be called for each attack class implemented in the attacks directory.
    The reason for this is to avoid circular imports and to allow dynamic discovery of attack classes
    upon system runtime, this was made to replace ENUM CLASS in the past since Python Enums are evaluated 
    once at import time and aren’t meant to mutate dynamically.
    
    ATTACK_REGISTRY = {
    "attack a": <Class A>,
    "attack b": <Class B>,
    ...
    }
    """
    def decorator(attack_class: Type) -> Type:
        """
        __DESCRIPTION__:

        Register an attack class, and run at import time of the module.
        ATTACK_REGISTRY["attack name"] = <Class NAME>
        """
        attack = class_name.lower()
        ATTACK_REGISTRY[class_name.lower()] = attack_class
        attack_class.__attack_type__ = attack
        return attack_class
    return decorator

def get_attack(name: str):
    """
    __DESCRIPTION__:
    Instantiate attack strategy by name (case-insensitive).

    __USAGE__:
    attack_instance = get_attack("attack name", arg1, arg2, kwarg1=value1)
    
    """
    cls = ATTACK_REGISTRY.get(name.lower())
    if cls is None:
        raise KeyError(f"No attack registered under name '{name}'")
    return cls


def autodiscover_attack_modules():
    """
    __DESCRIPTION__:
    Import all submodules under the package of node.attacks.
    Each attack module should import the register decorator and decorate its class.
    This function just ensures every file in this attacks/ dir is imported and for each attack file/class,
    it triggers the decorators in those modules to execute which then populates the registry automatically
    
    __USAGE__:
    # TODO check this
    call this for each simulator instance
    autodiscover_attack_modules()
    """
    #package = os.path.basename(os.path.dirname(__file__))
    package = __package__
    pkg = importlib.import_module(package)
    package_path = pkg.__path__  
    for finder, name, ispkg in pkgutil.iter_modules(package_path):
        full_name = f"{package}.{name}"
        importlib.import_module(full_name)