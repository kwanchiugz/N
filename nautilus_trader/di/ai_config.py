# -------------------------------------------------------------------------------------------------
#  Copyright (C) 2015-2025 Nautech Systems Pty Ltd. All rights reserved.
#  https://nautechsystems.io
#
#  Licensed under the GNU Lesser General Public License Version 3.0 (the "License");
#  You may not use this file except in compliance with the License.
#  You may obtain a copy of the License at https://www.gnu.org/licenses/lgpl-3.0.en.html
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
# -------------------------------------------------------------------------------------------------
"""
AI-enhanced DI container configuration.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
from enum import Enum


class AIRecommendationLevel(str, Enum):
    """AI recommendation confidence levels."""
    DISABLED = "disabled"
    LOW = "low"        # Only high-confidence recommendations
    MEDIUM = "medium"  # Moderate confidence threshold
    HIGH = "high"      # Show all recommendations


@dataclass
class AIContainerConfig:
    """
    Configuration for AI-enhanced DI container.
    
    This is a simplified, focused configuration compared to the enterprise version.
    """
    
    # Core AI features
    enable_ai_recommendations: bool = True
    recommendation_level: AIRecommendationLevel = AIRecommendationLevel.MEDIUM
    enable_auto_registration: bool = False
    
    # Performance monitoring (simplified)
    track_performance: bool = True
    performance_warning_threshold_ms: float = 2.0  # Warn if resolution > 2ms
    
    # Auto-registration safety
    auto_register_trusted_prefixes: List[str] = None
    auto_register_max_dependencies: int = 5
    
    # AI service integration
    ai_provider_config: Optional[Dict] = None
    enable_real_ai_recommendations: bool = False  # Use actual AI service vs heuristics
    
    # Usage analytics
    enable_usage_analytics: bool = True
    analytics_retention_hours: int = 24
    
    def __post_init__(self):
        if self.auto_register_trusted_prefixes is None:
            self.auto_register_trusted_prefixes = ["nautilus_trader"]
    
    def get_ai_service_config(self) -> Optional[Dict]:
        """Get AI service configuration if available."""
        if not self.enable_real_ai_recommendations or not self.ai_provider_config:
            return None
            
        return {
            "provider": self.ai_provider_config.get("provider", "deepseek"),
            "model": self.ai_provider_config.get("model", "deepseek-chat"),
            "temperature": 0.3,  # Low temperature for consistent recommendations
            "max_tokens": 1000,
        }
    
    def should_show_recommendation(self, confidence: float) -> bool:
        """Determine if recommendation should be shown based on confidence and level."""
        if self.recommendation_level == AIRecommendationLevel.DISABLED:
            return False
        elif self.recommendation_level == AIRecommendationLevel.LOW:
            return confidence >= 0.8
        elif self.recommendation_level == AIRecommendationLevel.MEDIUM:
            return confidence >= 0.6
        else:  # HIGH
            return confidence >= 0.4


class AIContainerBuilder:
    """
    Builder for creating AI-enhanced DI container with configuration.
    """
    
    def __init__(self):
        self.config = AIContainerConfig()
    
    def enable_ai(self, level: AIRecommendationLevel = AIRecommendationLevel.MEDIUM) -> "AIContainerBuilder":
        """Enable AI recommendations with specified level."""
        self.config.enable_ai_recommendations = True
        self.config.recommendation_level = level
        return self
    
    def enable_auto_registration(self, trusted_prefixes: List[str] = None) -> "AIContainerBuilder":
        """Enable auto-registration with trusted prefixes."""
        self.config.enable_auto_registration = True
        if trusted_prefixes:
            self.config.auto_register_trusted_prefixes = trusted_prefixes
        return self
    
    def with_ai_provider(self, provider: str, **kwargs) -> "AIContainerBuilder":
        """Configure real AI provider for enhanced recommendations."""
        self.config.enable_real_ai_recommendations = True
        self.config.ai_provider_config = {"provider": provider, **kwargs}
        return self
    
    def with_performance_monitoring(self, threshold_ms: float = 2.0) -> "AIContainerBuilder":
        """Enable performance monitoring with custom threshold."""
        self.config.track_performance = True
        self.config.performance_warning_threshold_ms = threshold_ms
        return self
    
    def build(self) -> "AIEnhancedDIContainer":
        """Build the container with current configuration."""
        from nautilus_trader.di.ai_enhanced_container import AIEnhancedDIContainer
        
        return AIEnhancedDIContainer(
            enable_ai_recommendations=self.config.enable_ai_recommendations,
            auto_register=self.config.enable_auto_registration,
        )


# Preset configurations for common scenarios
def create_trading_container() -> "AIEnhancedDIContainer":
    """Create AI-enhanced container optimized for trading applications."""
    return (AIContainerBuilder()
            .enable_ai(AIRecommendationLevel.MEDIUM)
            .enable_auto_registration(["nautilus_trader", "trading"])
            .with_performance_monitoring(1.0)  # Stricter performance for trading
            .build())


def create_development_container() -> "AIEnhancedDIContainer":
    """Create AI-enhanced container for development with verbose recommendations."""
    return (AIContainerBuilder()
            .enable_ai(AIRecommendationLevel.HIGH)
            .enable_auto_registration()
            .with_performance_monitoring(5.0)  # More relaxed for dev
            .build())


def create_production_container() -> "AIEnhancedDIContainer":
    """Create AI-enhanced container for production with minimal overhead."""
    return (AIContainerBuilder()
            .enable_ai(AIRecommendationLevel.LOW)  # Only high-confidence recommendations
            .with_performance_monitoring(0.5)  # Very strict for production
            .build())