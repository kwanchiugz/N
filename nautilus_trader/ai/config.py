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
AI module configuration.
"""

from __future__ import annotations

from nautilus_trader.common.config import NautilusConfig
from nautilus_trader.common.config import PositiveFloat
from nautilus_trader.common.config import PositiveInt


class AIProviderConfig(NautilusConfig, frozen=True):
    """
    Base configuration for AI providers.
    
    Parameters
    ----------
    api_key : str, optional
        The API key for the provider. If None, will be loaded from secure storage.
    base_url : str, optional
        The base URL for API requests.
    timeout : PositiveFloat, default 30.0
        Request timeout in seconds.
    max_retries : PositiveInt, default 3
        Maximum number of retry attempts.
    retry_delay : PositiveFloat, default 1.0
        Initial retry delay in seconds.
    
    """
    
    api_key: str | None = None
    base_url: str | None = None
    timeout: PositiveFloat = 30.0
    max_retries: PositiveInt = 3
    retry_delay: PositiveFloat = 1.0


class DeepSeekConfig(AIProviderConfig, frozen=True):
    """
    Configuration for DeepSeek AI provider.
    
    Parameters
    ----------
    model : str, default "deepseek-chat"
        The model to use for inference.
    temperature : float, default 0.7
        Sampling temperature (0.0 to 2.0).
    max_tokens : PositiveInt, default 4096
        Maximum tokens in response.
    top_p : float, default 0.95
        Nucleus sampling parameter.
    frequency_penalty : float, default 0.0
        Frequency penalty (-2.0 to 2.0).
    presence_penalty : float, default 0.0
        Presence penalty (-2.0 to 2.0).
    
    """
    
    model: str = "deepseek-chat"
    temperature: float = 0.7
    max_tokens: PositiveInt = 4096
    top_p: float = 0.95
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    
    def __post_init__(self):
        # Validate temperature
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError("temperature must be between 0.0 and 2.0")
        
        # Validate top_p
        if not 0.0 <= self.top_p <= 1.0:
            raise ValueError("top_p must be between 0.0 and 1.0")
            
        # Validate penalties
        if not -2.0 <= self.frequency_penalty <= 2.0:
            raise ValueError("frequency_penalty must be between -2.0 and 2.0")
        if not -2.0 <= self.presence_penalty <= 2.0:
            raise ValueError("presence_penalty must be between -2.0 and 2.0")


class AIAnalyzerConfig(NautilusConfig, frozen=True):
    """
    Configuration for AI analyzers.
    
    Parameters
    ----------
    provider_config : AIProviderConfig
        Configuration for the AI provider.
    cache_results : bool, default True
        Whether to cache analysis results.
    cache_ttl : PositiveInt, default 300
        Cache time-to-live in seconds.
    rate_limit_calls : PositiveInt, default 100
        Maximum API calls per period.
    rate_limit_period : PositiveInt, default 60
        Rate limit period in seconds.
    
    """
    
    provider_config: AIProviderConfig
    cache_results: bool = True
    cache_ttl: PositiveInt = 300
    rate_limit_calls: PositiveInt = 100
    rate_limit_period: PositiveInt = 60


class AIConfig(NautilusConfig, frozen=True):
    """
    Main AI module configuration.
    
    Parameters
    ----------
    provider : str, default "deepseek"
        The AI provider to use.
    provider_config : AIProviderConfig
        Provider-specific configuration.
    analyzer_config : AIAnalyzerConfig
        Analyzer configuration.
    enable_market_analysis : bool, default True
        Enable AI market analysis.
    enable_risk_assessment : bool, default True
        Enable AI risk assessment.
    enable_signal_generation : bool, default True
        Enable AI signal generation.
    log_prompts : bool, default False
        Whether to log AI prompts (be careful with sensitive data).
    
    """
    
    provider: str = "deepseek"
    provider_config: AIProviderConfig
    analyzer_config: AIAnalyzerConfig
    enable_market_analysis: bool = True
    enable_risk_assessment: bool = True
    enable_signal_generation: bool = True
    log_prompts: bool = False