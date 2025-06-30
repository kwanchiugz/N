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
AI-Enhanced Dependency Injection Framework for Nautilus Trader.

Features:
- Intelligent service lifecycle management with AI pattern recognition
- Auto-registration with safety mechanisms
- Performance monitoring and optimization suggestions
- Advanced caching and resilience patterns
"""

from nautilus_trader.di.container import (
    DIContainer,
    Injectable,
    Singleton,
    Transient,
    Scoped,
    inject,
)
from nautilus_trader.di.ai_enhanced_container import (
    AIEnhancedDIContainer,
    AIServiceRecommendation,
)
from nautilus_trader.di.ai_config import (
    AIContainerConfig,
    AIRecommendationLevel,
    AIContainerBuilder,
    create_trading_container,
    create_development_container,
    create_production_container,
)
from nautilus_trader.di.ai_service_discovery import (
    AIServiceDiscovery,
    ServiceCandidate,
    DependencyGraph,
    create_ai_discovery_session,
    discover_and_recommend,
)
from nautilus_trader.di.ai_integration import (
    ServiceOptimizationRequest,
    OptimizationSuggestion,
    AIContainerOptimizer,
    AIContainerIntegration,
)
from nautilus_trader.di.providers import (
    Provider,
    SingletonProvider,
    TransientProvider,
    ScopedProvider,
    FactoryProvider,
)
from nautilus_trader.di.registry import ServiceRegistry
from nautilus_trader.di.bootstrap import Bootstrap


__all__ = [
    # Core Container
    "DIContainer",
    "Injectable",
    "Singleton",
    "Transient",
    "Scoped",
    "inject",
    # AI-Enhanced Components
    "AIEnhancedDIContainer",
    "AIServiceRecommendation",
    "AIContainerConfig",
    "AIRecommendationLevel",
    "AIContainerBuilder",
    "create_trading_container",
    "create_development_container",
    "create_production_container",
    "AIServiceDiscovery",
    "ServiceCandidate",
    "DependencyGraph",
    "create_ai_discovery_session",
    "discover_and_recommend",
    "ServiceOptimizationRequest",
    "OptimizationSuggestion",
    "AIContainerOptimizer",
    "AIContainerIntegration",
    # Providers
    "Provider",
    "SingletonProvider",
    "TransientProvider",
    "ScopedProvider",
    "FactoryProvider",
    # Registry
    "ServiceRegistry",
    # Bootstrap
    "Bootstrap",
]