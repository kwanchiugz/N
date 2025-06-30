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
AI-powered portfolio optimization using advanced techniques.
"""

from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
from datetime import datetime
from decimal import Decimal
import numpy as np
import pandas as pd

from nautilus_trader.ai.analyzers.base import BaseAIAnalyzer
from nautilus_trader.ai.config import AIAnalyzerConfig
from nautilus_trader.ai.utils.prompts import PortfolioOptimizationPrompts
from nautilus_trader.model.position import Position
from nautilus_trader.model.identifiers import InstrumentId


class OptimizationObjective(str, Enum):
    """Portfolio optimization objectives."""
    MAX_SHARPE = "max_sharpe"
    MIN_RISK = "min_risk"
    MAX_RETURN = "max_return"
    RISK_PARITY = "risk_parity"
    MAX_DIVERSIFICATION = "max_diversification"


class PortfolioConstraints:
    """
    Constraints for portfolio optimization.
    
    Parameters
    ----------
    max_position_size : float
        Maximum position size as percentage (0-1)
    min_position_size : float
        Minimum position size as percentage
    max_leverage : float
        Maximum total leverage
    max_concentration : float
        Maximum single asset concentration
    sector_limits : dict, optional
        Maximum exposure per sector
    long_only : bool
        Whether to allow only long positions
    """
    
    def __init__(
        self,
        max_position_size: float = 0.3,
        min_position_size: float = 0.01,
        max_leverage: float = 1.0,
        max_concentration: float = 0.5,
        sector_limits: Optional[Dict[str, float]] = None,
        long_only: bool = False,
    ) -> None:
        self.max_position_size = max_position_size
        self.min_position_size = min_position_size
        self.max_leverage = max_leverage
        self.max_concentration = max_concentration
        self.sector_limits = sector_limits or {}
        self.long_only = long_only
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for AI prompts."""
        return {
            "max_position_size": self.max_position_size,
            "min_position_size": self.min_position_size,
            "max_leverage": self.max_leverage,
            "max_concentration": self.max_concentration,
            "sector_limits": self.sector_limits,
            "long_only": self.long_only,
        }


class PortfolioOptimizer(BaseAIAnalyzer):
    """
    AI-powered portfolio optimizer.
    
    This optimizer uses AI to:
    - Analyze market conditions and correlations
    - Predict returns and risks
    - Optimize portfolio weights
    - Consider transaction costs and constraints
    - Provide rebalancing recommendations
    """
    
    def __init__(
        self,
        config: AIAnalyzerConfig,
        cache: Optional[Any] = None,
    ) -> None:
        """Initialize portfolio optimizer."""
        super().__init__(config, cache)
        self.prompts = PortfolioOptimizationPrompts()
        
    async def optimize_portfolio(
        self,
        instruments: List[InstrumentId],
        historical_data: pd.DataFrame,
        objective: OptimizationObjective,
        constraints: PortfolioConstraints,
        current_positions: Optional[Dict[InstrumentId, Position]] = None,
        market_conditions: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Optimize portfolio allocation using AI.
        
        Parameters
        ----------
        instruments : List[InstrumentId]
            Instruments to include
        historical_data : pd.DataFrame
            Historical price data
        objective : OptimizationObjective
            Optimization objective
        constraints : PortfolioConstraints
            Portfolio constraints
        current_positions : Dict[InstrumentId, Position], optional
            Current portfolio positions
        market_conditions : Dict[str, Any], optional
            Current market conditions
            
        Returns
        -------
        Dict[str, Any]
            Optimization results including weights and metrics
        """
        # Calculate basic statistics
        returns = historical_data.pct_change().dropna()
        
        statistics = {
            "returns": {
                str(inst): {
                    "mean": float(returns[str(inst)].mean()),
                    "std": float(returns[str(inst)].std()),
                    "sharpe": float(returns[str(inst)].mean() / returns[str(inst)].std() * np.sqrt(252))
                    if returns[str(inst)].std() > 0 else 0,
                }
                for inst in instruments
            },
            "correlation_matrix": returns.corr().to_dict(),
            "covariance_matrix": returns.cov().to_dict(),
        }
        
        # Prepare current portfolio
        current_weights = {}
        if current_positions:
            total_value = sum(
                pos.quantity * pos.last_px for pos in current_positions.values()
            )
            
            for inst, pos in current_positions.items():
                weight = float(pos.quantity * pos.last_px / total_value) if total_value > 0 else 0
                current_weights[str(inst)] = weight
                
        # Build optimization prompt
        prompt = self.prompts.build_portfolio_optimization_prompt(
            instruments=[str(inst) for inst in instruments],
            statistics=statistics,
            objective=objective.value,
            constraints=constraints.to_dict(),
            current_weights=current_weights,
            market_conditions=market_conditions,
        )
        
        # Get AI optimization
        response = await self._provider.complete(
            prompt=prompt,
            system_prompt=self.prompts.PORTFOLIO_SYSTEM_PROMPT,
            temperature=0.2,  # Low temperature for consistent optimization
        )
        
        # Parse response
        result = self._parse_json_response(response.content)
        
        # Validate and normalize weights
        weights = result.get("weights", {})
        weights = self._validate_weights(weights, constraints)
        
        # Calculate portfolio metrics
        metrics = self._calculate_portfolio_metrics(
            weights,
            statistics,
            objective,
        )
        
        # Add metadata
        result.update({
            "weights": weights,
            "metrics": metrics,
            "timestamp": datetime.utcnow().isoformat(),
            "objective": objective.value,
            "model": response.model,
        })
        
        return result
        
    async def analyze_rebalancing_needs(
        self,
        current_positions: Dict[InstrumentId, Position],
        target_weights: Dict[str, float],
        current_prices: Dict[InstrumentId, Decimal],
        transaction_costs: float = 0.001,  # 0.1%
        rebalancing_threshold: float = 0.05,  # 5% deviation
    ) -> Dict[str, Any]:
        """
        Analyze if portfolio needs rebalancing.
        
        Parameters
        ----------
        current_positions : Dict[InstrumentId, Position]
            Current positions
        target_weights : Dict[str, float]
            Target portfolio weights
        current_prices : Dict[InstrumentId, Decimal]
            Current market prices
        transaction_costs : float
            Transaction cost percentage
        rebalancing_threshold : float
            Minimum deviation to trigger rebalancing
            
        Returns
        -------
        Dict[str, Any]
            Rebalancing analysis and recommendations
        """
        # Calculate current weights
        total_value = Decimal("0")
        position_values = {}
        
        for inst_id, pos in current_positions.items():
            value = pos.quantity * current_prices[inst_id]
            position_values[inst_id] = value
            total_value += value
            
        current_weights = {
            str(inst_id): float(value / total_value) if total_value > 0 else 0
            for inst_id, value in position_values.items()
        }
        
        # Calculate deviations
        deviations = {}
        trades_needed = {}
        
        for inst_str, target_weight in target_weights.items():
            current_weight = current_weights.get(inst_str, 0)
            deviation = abs(target_weight - current_weight)
            deviations[inst_str] = deviation
            
            if deviation > rebalancing_threshold:
                # Calculate trade size needed
                target_value = float(total_value) * target_weight
                current_value = float(total_value) * current_weight
                trade_value = target_value - current_value
                trades_needed[inst_str] = trade_value
                
        # Estimate costs
        total_trade_value = sum(abs(v) for v in trades_needed.values())
        estimated_costs = total_trade_value * transaction_costs
        
        # Build analysis prompt
        prompt = self.prompts.build_rebalancing_analysis_prompt(
            current_weights=current_weights,
            target_weights=target_weights,
            deviations=deviations,
            trades_needed=trades_needed,
            transaction_costs=transaction_costs,
            estimated_costs=estimated_costs,
        )
        
        # Get AI analysis
        response = await self._provider.complete(
            prompt=prompt,
            system_prompt=self.prompts.PORTFOLIO_SYSTEM_PROMPT,
            temperature=0.3,
        )
        
        # Parse response
        analysis = self._parse_json_response(response.content)
        
        # Add calculated data
        analysis.update({
            "current_weights": current_weights,
            "target_weights": target_weights,
            "deviations": deviations,
            "trades_needed": trades_needed,
            "estimated_costs": estimated_costs,
            "needs_rebalancing": len(trades_needed) > 0,
            "total_portfolio_value": float(total_value),
        })
        
        return analysis
        
    async def generate_allocation_strategy(
        self,
        capital: Decimal,
        instruments: List[InstrumentId],
        risk_profile: str,  # "conservative", "moderate", "aggressive"
        investment_horizon: str,  # "short", "medium", "long"
        market_outlook: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Generate allocation strategy based on investor profile.
        
        Parameters
        ----------
        capital : Decimal
            Available capital
        instruments : List[InstrumentId]
            Available instruments
        risk_profile : str
            Investor risk profile
        investment_horizon : str
            Investment time horizon
        market_outlook : Dict[str, Any], optional
            Market outlook and predictions
            
        Returns
        -------
        Dict[str, Any]
            Recommended allocation strategy
        """
        # Build strategy prompt
        prompt = self.prompts.build_allocation_strategy_prompt(
            capital=str(capital),
            instruments=[str(inst) for inst in instruments],
            risk_profile=risk_profile,
            investment_horizon=investment_horizon,
            market_outlook=market_outlook,
        )
        
        # Get AI strategy
        response = await self._provider.complete(
            prompt=prompt,
            system_prompt=self.prompts.PORTFOLIO_SYSTEM_PROMPT,
            temperature=0.4,
        )
        
        # Parse response
        strategy = self._parse_json_response(response.content)
        
        # Add metadata
        strategy.update({
            "capital": str(capital),
            "risk_profile": risk_profile,
            "investment_horizon": investment_horizon,
            "timestamp": datetime.utcnow().isoformat(),
            "model": response.model,
        })
        
        return strategy
        
    def _validate_weights(
        self,
        weights: Dict[str, float],
        constraints: PortfolioConstraints,
    ) -> Dict[str, float]:
        """Validate and normalize portfolio weights."""
        # Remove positions below minimum
        weights = {
            k: v for k, v in weights.items()
            if v >= constraints.min_position_size
        }
        
        # Cap positions at maximum
        weights = {
            k: min(v, constraints.max_position_size)
            for k, v in weights.items()
        }
        
        # Normalize to sum to 1
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}
            
        # Check leverage constraint
        if constraints.max_leverage < 1.0:
            # Scale down if needed
            scale = min(1.0, constraints.max_leverage)
            weights = {k: v * scale for k, v in weights.items()}
            
        return weights
        
    def _calculate_portfolio_metrics(
        self,
        weights: Dict[str, float],
        statistics: Dict[str, Any],
        objective: OptimizationObjective,
    ) -> Dict[str, float]:
        """Calculate portfolio performance metrics."""
        # Extract returns statistics
        returns_data = statistics["returns"]
        correlation_matrix = statistics["correlation_matrix"]
        
        # Calculate weighted returns
        portfolio_return = sum(
            weights.get(inst, 0) * data["mean"]
            for inst, data in returns_data.items()
        )
        
        # Calculate portfolio variance
        portfolio_variance = 0.0
        instruments = list(weights.keys())
        
        for i, inst1 in enumerate(instruments):
            for j, inst2 in enumerate(instruments):
                w1 = weights.get(inst1, 0)
                w2 = weights.get(inst2, 0)
                
                if inst1 in correlation_matrix and inst2 in correlation_matrix[inst1]:
                    corr = correlation_matrix[inst1][inst2]
                    std1 = returns_data[inst1]["std"]
                    std2 = returns_data[inst2]["std"]
                    
                    portfolio_variance += w1 * w2 * corr * std1 * std2
                    
        portfolio_std = np.sqrt(portfolio_variance)
        
        # Calculate metrics
        sharpe_ratio = (portfolio_return / portfolio_std * np.sqrt(252)) if portfolio_std > 0 else 0
        
        # Calculate diversification ratio
        weighted_avg_std = sum(
            weights.get(inst, 0) * returns_data[inst]["std"]
            for inst in instruments
        )
        diversification_ratio = weighted_avg_std / portfolio_std if portfolio_std > 0 else 1
        
        return {
            "expected_return": portfolio_return * 252,  # Annualized
            "volatility": portfolio_std * np.sqrt(252),  # Annualized
            "sharpe_ratio": sharpe_ratio,
            "diversification_ratio": diversification_ratio,
            "max_weight": max(weights.values()) if weights else 0,
            "min_weight": min(weights.values()) if weights else 0,
            "num_positions": len([w for w in weights.values() if w > 0]),
        }