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
Advanced AI capabilities for Nautilus Trader.
"""

from nautilus_trader.ai.advanced.portfolio_optimizer import (
    PortfolioOptimizer,
    OptimizationObjective,
    PortfolioConstraints,
)
from nautilus_trader.ai.advanced.market_regime import (
    MarketRegimeDetector,
    RegimeType,
    RegimeTransition,
)
from nautilus_trader.ai.advanced.sentiment_analyzer import (
    SentimentAnalyzer,
    SentimentSource,
    MarketSentiment,
)
from nautilus_trader.ai.advanced.anomaly_detector import (
    AnomalyDetector,
    AnomalyType,
    MarketAnomaly,
)


__all__ = [
    # Portfolio Optimization
    "PortfolioOptimizer",
    "OptimizationObjective",
    "PortfolioConstraints",
    # Market Regime
    "MarketRegimeDetector",
    "RegimeType",
    "RegimeTransition",
    # Sentiment Analysis
    "SentimentAnalyzer",
    "SentimentSource",
    "MarketSentiment",
    # Anomaly Detection
    "AnomalyDetector",
    "AnomalyType",
    "MarketAnomaly",
]