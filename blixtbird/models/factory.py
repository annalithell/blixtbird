# blixtbird/models/factory.py

from typing import Dict, Type

from blixtbird.models.base import ModelBase
from blixtbird.models.cnn import Net


class ModelFactory:
    """
    Factory class for creating model instances.
    This makes it easy for users to select different models.
    """
    
    # Registry of available models
    _models: Dict[str, Type[ModelBase]] = {
        'cnn': Net,
        # 'mlp': MLP,
    }
    
    @classmethod
    def register_model(cls, name: str, model_class: Type[ModelBase]) -> None:
        """
        Register a new model type.
        
        Args:
            name: Name of the model
            model_class: Model class
        """
        cls._models[name] = model_class
    
    @classmethod
    def get_model(cls, model_name: str, **kwargs) -> ModelBase:
        """
        Get a model instance by name.
        
        Args:
            model_name: Name of the model
            **kwargs: Additional arguments to pass to the model constructor
            
        Returns:
            Instance of the requested model
            
        Raises:
            ValueError: If the model name is not recognized
        """
        if model_name not in cls._models:
            available_models = list(cls._models.keys())
            raise ValueError(f"Unknown model: '{model_name}'. Available models: {available_models}")
        
        model_class = cls._models[model_name]
        return model_class(**kwargs)
    
    @classmethod
    def list_available_models(cls) -> Dict[str, Type[ModelBase]]:
        """
        List all available models.
        
        Returns:
            Dictionary mapping model names to model classes
        """
        return cls._models.copy()
