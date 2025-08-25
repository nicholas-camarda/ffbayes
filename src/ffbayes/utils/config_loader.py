#!/usr/bin/env python3
"""
Configuration Loader - Centralized configuration management for FFBayes
Loads user preferences from config file and environment variables with fallbacks.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict


class ConfigLoader:
    """Centralized configuration loader for FFBayes."""
    
    def __init__(self, config_file: str = None):
        if config_file is None:
            from ffbayes.utils.path_constants import get_user_config_file
            config_file = str(get_user_config_file())
        """Initialize configuration loader."""
        self.config_file = Path(config_file)
        self.config = self._load_config()
        self._override_with_env_vars()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from JSON file."""
        if not self.config_file.exists():
            print(f"⚠️  Config file not found: {self.config_file}")
            print("   Using default configuration")
            return self._get_default_config()
        
        try:
            with open(self.config_file, 'r') as f:
                config = json.load(f)
            print(f"✅ Loaded configuration from: {self.config_file}")
            return config
        except Exception as e:
            print(f"❌ Error loading config file: {e}")
            print("   Using default configuration")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration values."""
        return {
            "league_settings": {
                "draft_position": 10,
                "league_size": 10,
                "scoring_type": "PPR",
                "ppr_value": 0.5,
                "risk_tolerance": "medium"
            },
            "vor_settings": {
                "ppr": 0.5,
                "top_rank": 120,
                "scraping_enabled": True,
                "fantasy_pros_source": True
            },
            "model_settings": {
                "monte_carlo_simulations": 5000,
                "historical_years": 5,
                "uncertainty_model": "random_forest",
                "confidence_level": 0.95
            },
            "output_settings": {
                "create_excel": True,
                "create_text": True,
                "create_visualizations": True,
                "output_format": "human_readable"
            },
            "data_settings": {
                "auto_update": True,
                "data_sources": ["nfl_official", "fantasy_pros", "pro_football_reference"],
                "cache_enabled": True,
                "validation_strict": True
            },
            "pipeline_settings": {
                "parallel_execution": True,
                "max_workers": 8,
                "timeout_default": 300,
                "retry_count": 2,
                "fail_fast": True
            }
        }
    
    def _override_with_env_vars(self):
        """Override config with environment variables (higher priority)."""
        env_mappings = {
            # League settings
            "DRAFT_POSITION": ("league_settings", "draft_position", int),
            "LEAGUE_SIZE": ("league_settings", "league_size", int),
            "RISK_TOLERANCE": ("league_settings", "risk_tolerance", str),
            
            # VOR settings
            "VOR_PPR": ("vor_settings", "ppr", float),
            "VOR_TOP_RANK": ("vor_settings", "top_rank", int),
            
            # Model settings
            "MC_SIMULATIONS": ("model_settings", "monte_carlo_simulations", int),
            "HISTORICAL_YEARS": ("model_settings", "historical_years", int),
            
            # Pipeline settings
            "MAX_WORKERS": ("pipeline_settings", "max_workers", int),
            "TIMEOUT_DEFAULT": ("pipeline_settings", "timeout_default", int),
        }
        
        for env_var, (section, key, type_func) in env_mappings.items():
            if env_var in os.environ:
                try:
                    value = type_func(os.environ[env_var])
                    self.config[section][key] = value
                    print(f"   🔧 {env_var}={value} (from environment)")
                except ValueError as e:
                    print(f"   ⚠️  Invalid {env_var} value: {os.environ[env_var]} ({e})")
    
    def get(self, section: str, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        try:
            return self.config[section][key]
        except KeyError:
            return default
    
    def get_league_setting(self, key: str, default: Any = None) -> Any:
        """Get league setting value."""
        return self.get("league_settings", key, default)
    
    def get_vor_setting(self, key: str, default: Any = None) -> Any:
        """Get VOR setting value."""
        return self.get("vor_settings", key, default)
    
    def get_model_setting(self, key: str, default: Any = None) -> Any:
        """Get model setting value."""
        return self.get("model_settings", key, default)
    
    def get_output_setting(self, key: str, default: Any = None) -> Any:
        """Get output setting value."""
        return self.get("output_settings", key, default)
    
    def get_pipeline_setting(self, key: str, default: Any = None) -> Any:
        """Get pipeline setting value."""
        return self.get("pipeline_settings", key, default)
    
    def get_all(self) -> Dict[str, Any]:
        """Get complete configuration."""
        return self.config.copy()
    
    def print_config(self):
        """Print current configuration."""
        print("\n" + "="*60)
        print("FFBayes Configuration")
        print("="*60)
        
        for section, settings in self.config.items():
            if isinstance(settings, dict):
                print(f"\n📋 {section.replace('_', ' ').title()}:")
                for key, value in settings.items():
                    print(f"   {key}: {value}")
            else:
                print(f"\n📋 {section.replace('_', ' ').title()}: {settings}")
        
        print("\n" + "="*60)


# Global configuration instance
config = ConfigLoader()


def get_config() -> ConfigLoader:
    """Get global configuration instance."""
    return config


if __name__ == "__main__":
    """Test configuration loading."""
    config.print_config()
