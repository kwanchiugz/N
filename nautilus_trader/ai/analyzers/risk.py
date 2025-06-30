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
AI-powered risk analyzer using DeepSeek API.
"""

import json
from decimal import Decimal
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from nautilus_trader.ai.analyzers.base import BaseAIAnalyzer
from nautilus_trader.ai.config import AIAnalyzerConfig
from nautilus_trader.ai.utils.prompts import RiskAnalysisPrompts
from nautilus_trader.ai.models.responses import (
    AIResponseAdapter,
    RiskAssessmentResponse,
    PortfolioRiskResponse,
)
from nautilus_trader.common.cache import Cache
from nautilus_trader.core.datetime import dt_to_unix_nanos
from nautilus_trader.model.position import Position
from nautilus_trader.model.data import Bar


class RiskAnalyzer(BaseAIAnalyzer):
    """
    AI-powered risk analyzer for trading positions and portfolios.
    
    This analyzer uses AI to:
    - Assess position-specific risks
    - Evaluate portfolio risk profiles
    - Identify correlation and concentration risks
    - Perform stress testing scenarios
    - Recommend risk mitigation strategies
    """
    
    def __init__(
        self,
        config: AIAnalyzerConfig,
        cache: Optional[Cache] = None,
    ) -> None:
        """Initialize the risk analyzer."""
        super().__init__(config, cache)
        self.prompts = RiskAnalysisPrompts()
        
    async def assess_position_risk(
        self,
        position: Position,
        current_price: Decimal,
        market_volatility: Optional[float] = None,
        recent_bars: Optional[List[Bar]] = None,
    ) -> Optional[RiskAssessmentResponse]:
        """
        Assess risk for a specific trading position.
        
        Parameters
        ----------
        position : Position
            The position to analyze
        current_price : Decimal
            Current market price
        market_volatility : float, optional
            Current market volatility (e.g., ATR)
        recent_bars : List[Bar], optional
            Recent price bars for context
            
        Returns
        -------
        RiskAssessmentResponse or None
            Validated risk assessment including risk score, factors, and recommendations
        """
        # Prepare position details
        position_details = {
            "instrument_id": str(position.instrument_id),
            "side": "long" if position.is_long else "short",
            "quantity": str(position.quantity),
            "entry_price": str(position.avg_px_open),
            "current_price": str(current_price),
            "unrealized_pnl": str(position.unrealized_pnl(current_price)),
            "duration_minutes": (dt_to_unix_nanos(self._clock.utc_now()) - 
                               dt_to_unix_nanos(position.ts_opened)) / 1e9 / 60,
        }
        
        # Calculate position metrics
        if position.is_long:
            price_change_pct = ((current_price - position.avg_px_open) / 
                               position.avg_px_open * 100)
        else:
            price_change_pct = ((position.avg_px_open - current_price) / 
                               position.avg_px_open * 100)
            
        position_details["price_change_pct"] = f"{price_change_pct:.2f}%"
        
        # Prepare market context
        market_context = {
            "volatility": market_volatility if market_volatility else "Unknown",
            "volatility_regime": self._classify_volatility(market_volatility),
        }
        
        # Add recent price action if available
        if recent_bars and len(recent_bars) >= 5:
            recent_prices = [float(bar.close) for bar in recent_bars[-5:]]
            market_context["recent_trend"] = "up" if recent_prices[-1] > recent_prices[0] else "down"
            market_context["price_momentum"] = f"{((recent_prices[-1] - recent_prices[0]) / recent_prices[0] * 100):.2f}%"
            
        # Check cache first
        cache_key = f"position_risk_{position.id}_{current_price}"
        cached_result = self._get_cached_result(cache_key)
        if cached_result and isinstance(cached_result, RiskAssessmentResponse):
            return cached_result
            
        # Build prompt
        prompt = self.prompts.build_position_risk_prompt(
            position_details=position_details,
            market_context=market_context,
        )
        
        # Get AI analysis
        try:
            response = await self._provider.complete(
                prompt=prompt,
                system_prompt=self.prompts.SYSTEM_PROMPT,
                temperature=0.3,  # Lower temperature for consistent risk assessment
            )
            
            # Parse and validate response using Pydantic
            risk_assessment = AIResponseAdapter.safe_parse(
                response_type="risk_assessment",
                raw_response=response.content,
            )
            
            if not risk_assessment:
                self._logger.error("Failed to parse risk assessment response")
                return None
                
            # Cache result
            self._cache_result(cache_key, risk_assessment)
            
            # Log high-risk positions
            if risk_assessment.risk_score > 0.7:
                self._logger.warning(
                    f"High risk position detected: {position.id} "
                    f"(risk_score: {risk_assessment.risk_score})"
                )
                
            return risk_assessment
            
        except Exception as e:
            self._logger.error(f"Position risk assessment failed: {e}")
            # Return None instead of fallback to maintain type safety
            return None
            
    async def assess_portfolio_risk(
        self,
        positions: List[Position],
        current_prices: Dict[str, Decimal],
        correlation_matrix: Optional[pd.DataFrame] = None,
        account_balance: Optional[Decimal] = None,
    ) -> Optional[PortfolioRiskResponse]:
        """
        Assess risk for entire portfolio.
        
        Parameters
        ----------
        positions : List[Position]
            List of open positions
        current_prices : Dict[str, Decimal]
            Current prices for each instrument
        correlation_matrix : pd.DataFrame, optional
            Correlation matrix between instruments
        account_balance : Decimal, optional
            Total account balance for risk calculations
            
        Returns
        -------
        PortfolioRiskResponse or None
            Validated portfolio risk assessment including diversification and concentration risks
        """
        if not positions:
            # Return None for empty portfolio
            return None
            
        # Calculate portfolio summary
        portfolio_summary = self._calculate_portfolio_summary(
            positions, current_prices, account_balance
        )
        
        # Prepare correlation data
        correlation_data = None
        if correlation_matrix is not None:
            correlation_data = {
                "average_correlation": float(correlation_matrix.mean().mean()),
                "max_correlation": float(correlation_matrix.max().max()),
                "highly_correlated_pairs": self._find_high_correlations(correlation_matrix),
            }
            
        # Check cache
        positions_str = "_".join(str(p.id) for p in positions)
        cache_key = f"portfolio_risk_{positions_str}_{hash(str(current_prices))}"
        cached_result = self._get_cached_result(cache_key)
        if cached_result and isinstance(cached_result, PortfolioRiskResponse):
            return cached_result
            
        # Build prompt
        prompt = self.prompts.build_portfolio_risk_prompt(
            portfolio_summary=portfolio_summary,
            correlation_matrix=correlation_data,
        )
        
        # Get AI analysis
        try:
            response = await self._provider.complete(
                prompt=prompt,
                system_prompt=self.prompts.SYSTEM_PROMPT,
                temperature=0.3,
            )
            
            # Parse and validate response using Pydantic
            portfolio_risk = AIResponseAdapter.safe_parse(
                response_type="portfolio_risk",
                raw_response=response.content,
            )
            
            if not portfolio_risk:
                self._logger.error("Failed to parse portfolio risk response")
                return None
                
            # Cache result
            self._cache_result(cache_key, portfolio_risk)
            
            return portfolio_risk
            
        except Exception as e:
            self._logger.error(f"Portfolio risk assessment failed: {e}")
            # Return None instead of fallback to maintain type safety
            return None
            
    async def calculate_var(
        self,
        positions: List[Position],
        current_prices: Dict[str, Decimal],
        confidence_level: float = 0.95,
        time_horizon_days: int = 1,
        historical_returns: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """
        Calculate Value at Risk (VaR) for portfolio.
        
        Parameters
        ----------
        positions : List[Position]
            List of positions
        current_prices : Dict[str, Decimal]
            Current prices
        confidence_level : float
            Confidence level (e.g., 0.95 for 95% VaR)
        time_horizon_days : int
            Time horizon in days
        historical_returns : pd.DataFrame, optional
            Historical returns data
            
        Returns
        -------
        Dict[str, Any]
            VaR calculations and risk metrics
        """
        # Calculate portfolio value
        portfolio_value = Decimal("0")
        position_values = {}
        
        for position in positions:
            instrument_id = str(position.instrument_id)
            current_price = current_prices.get(instrument_id)
            
            if current_price:
                position_value = position.quantity * current_price
                position_values[instrument_id] = position_value
                portfolio_value += position_value
                
        if portfolio_value == 0:
            return {"error": "Cannot calculate VaR for zero-value portfolio"}
            
        # Use historical method if data available
        if historical_returns is not None and not historical_returns.empty:
            # Calculate portfolio returns
            weights = {
                inst: float(value / portfolio_value) 
                for inst, value in position_values.items()
            }
            
            portfolio_returns = sum(
                historical_returns.get(inst, 0) * weight 
                for inst, weight in weights.items()
            )
            
            # Calculate VaR
            var_percentile = (1 - confidence_level) * 100
            var_value = np.percentile(portfolio_returns, var_percentile)
            var_amount = float(portfolio_value) * abs(var_value) * np.sqrt(time_horizon_days)
            
            return {
                "var_95": var_amount,
                "var_pct": abs(var_value) * 100,
                "confidence_level": confidence_level,
                "time_horizon_days": time_horizon_days,
                "portfolio_value": float(portfolio_value),
                "method": "historical",
            }
        else:
            # Use parametric method with assumed volatility
            # This is a simplified calculation
            assumed_volatility = 0.02  # 2% daily volatility
            z_score = 1.645 if confidence_level == 0.95 else 2.326  # 95% or 99%
            
            var_amount = float(portfolio_value) * assumed_volatility * z_score * np.sqrt(time_horizon_days)
            
            return {
                "var_95": var_amount,
                "var_pct": assumed_volatility * z_score * 100,
                "confidence_level": confidence_level,
                "time_horizon_days": time_horizon_days,
                "portfolio_value": float(portfolio_value),
                "method": "parametric",
                "note": "Using assumed volatility due to lack of historical data",
            }
            
    def _classify_volatility(self, volatility: Optional[float]) -> str:
        """Classify volatility regime."""
        if volatility is None:
            return "unknown"
        elif volatility < 0.005:  # 0.5%
            return "very_low"
        elif volatility < 0.01:   # 1%
            return "low"
        elif volatility < 0.02:   # 2%
            return "normal"
        elif volatility < 0.03:   # 3%
            return "high"
        else:
            return "extreme"
            
    def _calculate_portfolio_summary(
        self,
        positions: List[Position],
        current_prices: Dict[str, Decimal],
        account_balance: Optional[Decimal],
    ) -> Dict[str, Any]:
        """Calculate portfolio summary statistics."""
        total_value = Decimal("0")
        position_values = []
        position_details = []
        
        for position in positions:
            instrument_id = str(position.instrument_id)
            current_price = current_prices.get(instrument_id, position.avg_px_open)
            
            position_value = position.quantity * current_price
            total_value += position_value
            position_values.append(position_value)
            
            position_details.append({
                "instrument": instrument_id,
                "side": "long" if position.is_long else "short",
                "value": str(position_value),
                "pnl": str(position.unrealized_pnl(current_price)),
            })
            
        # Calculate concentration
        if position_values:
            largest_position = max(position_values)
            largest_position_pct = float(largest_position / total_value) if total_value > 0 else 0
        else:
            largest_position_pct = 0
            
        summary = {
            "total_positions": len(positions),
            "total_exposure": str(total_value),
            "average_position_size": str(total_value / len(positions)) if positions else "0",
            "largest_position_pct": f"{largest_position_pct:.2%}",
            "positions": position_details,
        }
        
        if account_balance:
            exposure_pct = float(total_value / account_balance) if account_balance > 0 else 0
            summary["exposure_vs_balance"] = f"{exposure_pct:.2%}"
            
        return summary
        
    def _find_high_correlations(
        self,
        correlation_matrix: pd.DataFrame,
        threshold: float = 0.7,
    ) -> List[Dict[str, Any]]:
        """Find highly correlated instrument pairs."""
        high_correlations = []
        
        # Get upper triangle of correlation matrix
        for i in range(len(correlation_matrix.columns)):
            for j in range(i + 1, len(correlation_matrix.columns)):
                corr = correlation_matrix.iloc[i, j]
                
                if abs(corr) > threshold:
                    high_correlations.append({
                        "pair": f"{correlation_matrix.columns[i]}-{correlation_matrix.columns[j]}",
                        "correlation": float(corr),
                    })
                    
        return high_correlations
        
    def _get_fallback_position_risk(
        self,
        position: Position,
        current_price: Decimal,
        price_change_pct: float,
    ) -> Dict[str, Any]:
        """Fallback risk assessment using rules-based approach."""
        risk_score = 0.0
        risk_factors = []
        
        # Price movement risk
        if abs(price_change_pct) > 5:
            risk_score += 0.3
            risk_factors.append({
                "factor": "Large price movement",
                "severity": "high" if abs(price_change_pct) > 10 else "medium",
                "description": f"Position moved {price_change_pct:.2f}% from entry",
            })
            
        # Position duration risk
        duration_hours = (dt_to_unix_nanos(self._clock.utc_now()) - 
                         dt_to_unix_nanos(position.ts_opened)) / 1e9 / 3600
                         
        if duration_hours > 24:
            risk_score += 0.1
            risk_factors.append({
                "factor": "Extended holding period",
                "severity": "medium",
                "description": f"Position open for {duration_hours:.1f} hours",
            })
            
        # Loss risk
        if price_change_pct < -2:
            risk_score += 0.2
            risk_factors.append({
                "factor": "Significant loss",
                "severity": "high",
                "description": f"Position showing {price_change_pct:.2f}% loss",
            })
            
        # Normalize risk score
        risk_score = min(risk_score, 1.0)
        
        return {
            "risk_score": risk_score,
            "risk_factors": risk_factors,
            "recommendations": self._get_risk_recommendations(risk_score, risk_factors),
            "method": "fallback_rules_based",
        }
        
    def _get_fallback_portfolio_risk(
        self,
        portfolio_summary: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Fallback portfolio risk assessment."""
        # Simple concentration check
        largest_position_pct = float(portfolio_summary["largest_position_pct"].strip("%")) / 100
        
        concentration_score = 0.0
        if largest_position_pct > 0.5:
            concentration_score = 0.8
        elif largest_position_pct > 0.3:
            concentration_score = 0.5
        elif largest_position_pct > 0.2:
            concentration_score = 0.3
            
        return {
            "portfolio_risk_score": concentration_score,
            "diversification_score": 1 - concentration_score,
            "concentration_risks": [
                f"Largest position represents {portfolio_summary['largest_position_pct']} of portfolio"
            ] if concentration_score > 0.3 else [],
            "recommendations": [
                "Consider diversifying positions" if concentration_score > 0.5 else "Portfolio reasonably diversified"
            ],
            "method": "fallback_rules_based",
        }
        
    def _get_risk_recommendations(
        self,
        risk_score: float,
        risk_factors: List[Dict[str, Any]],
    ) -> List[str]:
        """Generate risk recommendations based on score and factors."""
        recommendations = []
        
        if risk_score > 0.7:
            recommendations.append("Consider reducing position size or closing position")
        elif risk_score > 0.5:
            recommendations.append("Monitor position closely and tighten stop loss")
            
        # Factor-specific recommendations
        for factor in risk_factors:
            if "price movement" in factor["factor"].lower():
                recommendations.append("Review stop loss levels")
            elif "loss" in factor["factor"].lower():
                recommendations.append("Consider cutting losses if trend continues")
            elif "holding period" in factor["factor"].lower():
                recommendations.append("Review position thesis and consider rebalancing")
                
        return recommendations