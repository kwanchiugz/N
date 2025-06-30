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
AI prompt templates for market and risk analysis.
"""

import json
from typing import Any, Dict, List, Optional


class MarketAnalysisPrompts:
    """Prompt templates for market analysis."""
    
    SYSTEM_PROMPT = """You are an expert quantitative trader and market analyst with deep knowledge of technical analysis, market microstructure, and trading strategies. 

Your analysis should be:
1. Data-driven and objective
2. Based on proven technical analysis principles
3. Considerate of market context and conditions
4. Clear about confidence levels and uncertainty

Always provide analysis in structured JSON format when requested."""
    
    def build_trend_analysis_prompt(
        self,
        instrument_id: str,
        market_data: List[Dict[str, Any]],
        additional_context: Optional[str] = None,
    ) -> str:
        """Build prompt for trend analysis."""
        prompt = f"""Analyze the market trend for {instrument_id} based on the following price data:

Recent Price Action (last {len(market_data)} bars):
{json.dumps(market_data[-10:], indent=2)}  # Show last 10 for context

Key Statistics:
- Current Price: {market_data[-1]['close']}
- 20-bar High: {max(bar['high'] for bar in market_data[-20:])}
- 20-bar Low: {min(bar['low'] for bar in market_data[-20:])}
- Average Volume: {sum(bar['volume'] for bar in market_data[-20:]) / 20}

{f"Additional Context: {additional_context}" if additional_context else ""}

Please analyze the trend and provide your assessment in the following JSON format:
{{
    "direction": "bullish|bearish|neutral",
    "strength": 0.0-1.0,  // 0=very weak, 1=very strong
    "confidence": 0.0-1.0,  // Your confidence in this analysis
    "support_levels": [price1, price2, ...],  // Key support levels
    "resistance_levels": [price1, price2, ...],  // Key resistance levels
    "reasoning": "Detailed explanation of your analysis"
}}"""
        return prompt
        
    def build_signal_generation_prompt(
        self,
        instrument_id: str,
        context: Dict[str, Any],
    ) -> str:
        """Build prompt for signal generation."""
        prompt = f"""Based on the current market analysis for {instrument_id}, evaluate whether to generate a trading signal:

Current Market Context:
- Price: {context['current_price']}
- Trend: {context['trend']} (strength: {context['trend_strength']})
- Support Levels: {context['support_levels']}
- Resistance Levels: {context['resistance_levels']}
- Risk Tolerance: {context['risk_tolerance']*100}%

Please evaluate and provide your recommendation in the following JSON format:
{{
    "generate_signal": true|false,
    "side": "buy|sell",  // Only if generate_signal is true
    "confidence": 0.0-1.0,
    "entry_price": price,
    "stop_loss": price,
    "take_profit": price,
    "reasoning": "Explanation of your recommendation",
    "timeframe": "1m|5m|15m|1h|4h|1d"  // Suggested holding period
}}

Important: Only generate a signal if you have high confidence (>0.7) and clear risk/reward setup."""
        return prompt
        
    def build_pattern_recognition_prompt(
        self,
        instrument_id: str,
        market_data: List[Dict[str, Any]],
    ) -> str:
        """Build prompt for pattern recognition."""
        prompt = f"""Analyze the price action for {instrument_id} and identify any significant technical patterns:

Price Data (last {len(market_data)} bars):
{json.dumps(market_data[-20:], indent=2)}  # Show last 20 for pattern context

Please identify any technical patterns and provide your analysis in the following JSON format:
{{
    "patterns": [
        {{
            "name": "Pattern name (e.g., Head and Shoulders, Triangle, Flag)",
            "type": "continuation|reversal",
            "confidence": 0.0-1.0,
            "target_price": price or null,
            "description": "Brief description of the pattern and its implications"
        }},
        ...
    ]
}}

Focus on high-probability patterns with clear structure. Include only patterns with confidence > 0.6."""
        return prompt
        
    def build_volatility_analysis_prompt(
        self,
        instrument_id: str,
        context: Dict[str, Any],
    ) -> str:
        """Build prompt for volatility analysis."""
        prompt = f"""Analyze the volatility characteristics for {instrument_id}:

Current Volatility Metrics:
- Historical Volatility (annualized): {context['historical_volatility']:.2%}
- Recent Price Action: {json.dumps(context['recent_price_action'][-5:], indent=2)}
{f"- Spread Analysis: {json.dumps(context['spread_analysis'], indent=2)}" if context.get('spread_analysis') else ""}

Please provide volatility analysis in the following JSON format:
{{
    "predicted_volatility": "Expected volatility over next period",
    "regime": "low|normal|high",
    "trend": "increasing|stable|decreasing",
    "risk_assessment": "Brief risk assessment based on volatility",
    "trading_implications": "How this volatility affects trading decisions"
}}"""
        return prompt


class RiskAnalysisPrompts:
    """Prompt templates for risk analysis."""
    
    SYSTEM_PROMPT = """You are an expert risk manager specializing in algorithmic trading. Your role is to:

1. Assess risk levels objectively
2. Identify potential threats to portfolio performance
3. Recommend appropriate risk mitigation strategies
4. Consider both systematic and idiosyncratic risks

Provide clear, actionable risk assessments in structured format."""
    
    def build_position_risk_prompt(
        self,
        position_details: Dict[str, Any],
        market_context: Dict[str, Any],
    ) -> str:
        """Build prompt for position risk assessment."""
        prompt = f"""Assess the risk for the following trading position:

Position Details:
{json.dumps(position_details, indent=2)}

Current Market Context:
{json.dumps(market_context, indent=2)}

Please provide risk assessment in the following JSON format:
{{
    "risk_score": 0.0-1.0,  // Overall risk level
    "risk_factors": [
        {{
            "factor": "Risk factor name",
            "severity": "low|medium|high",
            "description": "Explanation"
        }}
    ],
    "var_95": "95% Value at Risk estimate",
    "max_drawdown_expected": "Expected maximum drawdown",
    "recommendations": ["List of risk mitigation recommendations"],
    "stop_loss_adjustment": "Suggested stop loss level or null"
}}"""
        return prompt
        
    def build_portfolio_risk_prompt(
        self,
        portfolio_summary: Dict[str, Any],
        correlation_matrix: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Build prompt for portfolio risk assessment."""
        prompt = f"""Analyze the risk profile of the following portfolio:

Portfolio Summary:
{json.dumps(portfolio_summary, indent=2)}

{f"Correlation Matrix: {json.dumps(correlation_matrix, indent=2)}" if correlation_matrix else ""}

Please provide comprehensive risk analysis in the following JSON format:
{{
    "portfolio_risk_score": 0.0-1.0,
    "diversification_score": 0.0-1.0,
    "concentration_risks": ["List of concentration concerns"],
    "correlation_risks": ["List of correlation concerns"],
    "stress_test_results": {{
        "market_crash_10pct": "Expected portfolio impact",
        "volatility_spike": "Expected portfolio impact",
        "liquidity_crisis": "Expected portfolio impact"
    }},
    "recommendations": ["Risk management recommendations"],
    "optimal_hedge_suggestions": ["Hedging strategies if applicable"]
}}"""
        return prompt


class BasePrompts:
    """Base class for prompt templates."""
    
    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """Parse JSON response with error handling."""
        try:
            # Try to extract JSON from response
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                json_str = response.split("```")[1].split("```")[0].strip()
            else:
                json_str = response.strip()
                
            return json.loads(json_str)
        except Exception as e:
            # Return a basic response if parsing fails
            return {
                "error": f"Failed to parse response: {str(e)}",
                "raw_response": response,
            }


class StrategyAuditPrompts(BasePrompts):
    """Prompt templates for strategy auditing."""
    
    AUDIT_SYSTEM_PROMPT = """You are an expert trading strategy auditor specializing in quantitative trading systems.

Your expertise includes:
1. Risk management and position sizing
2. Entry/exit logic optimization
3. Parameter sensitivity analysis
4. Market regime adaptation
5. Execution quality and slippage

Provide detailed, actionable insights in structured JSON format."""
    
    def build_strategy_audit_prompt(
        self,
        strategy_name: str,
        code_summary: str,
        metrics: Dict[str, Any],
        issues: List[str],
    ) -> str:
        """Build strategy audit prompt."""
        return f"""Audit the trading strategy '{strategy_name}':

Strategy Code Summary:
{code_summary}

Performance Metrics:
{json.dumps(metrics, indent=2)}

Identified Issues:
{json.dumps(issues, indent=2)}

Provide comprehensive audit in JSON format:
{{
    "risk_assessment": {{
        "score": 0.0-1.0,
        "concerns": ["list of concerns"],
        "strengths": ["list of strengths"]
    }},
    "optimization_opportunities": [
        {{
            "area": "area name",
            "priority": "high|medium|low",
            "recommendation": "detailed recommendation",
            "expected_impact": "impact description"
        }}
    ],
    "market_regime_handling": {{
        "current_approach": "description",
        "weaknesses": ["list of weaknesses"],
        "recommendations": ["list of recommendations"]
    }},
    "overall_rating": "A-F",
    "summary": "executive summary"
}}"""


class PortfolioOptimizationPrompts(BasePrompts):
    """Prompt templates for portfolio optimization."""
    
    PORTFOLIO_SYSTEM_PROMPT = """You are a portfolio optimization expert with deep knowledge of:
1. Modern Portfolio Theory (MPT)
2. Risk parity and alternative risk premia
3. Factor-based investing
4. Dynamic asset allocation
5. Transaction cost optimization

Provide optimization recommendations based on quantitative analysis."""
    
    def build_portfolio_optimization_prompt(
        self,
        instruments: List[str],
        statistics: Dict[str, Any],
        objective: str,
        constraints: Dict[str, Any],
        current_weights: Dict[str, float],
        market_conditions: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Build portfolio optimization prompt."""
        return f"""Optimize portfolio allocation for the following instruments: {instruments}

Statistical Data:
- Returns & Risk: {json.dumps(statistics['returns'], indent=2)}
- Correlation Matrix: {json.dumps(statistics['correlation_matrix'], indent=2)}

Optimization Objective: {objective}
Constraints: {json.dumps(constraints, indent=2)}
Current Weights: {json.dumps(current_weights, indent=2)}
{f"Market Conditions: {json.dumps(market_conditions, indent=2)}" if market_conditions else ""}

Provide optimal allocation in JSON format:
{{
    "weights": {{
        "instrument1": weight,
        "instrument2": weight,
        ...
    }},
    "expected_metrics": {{
        "return": annual_return,
        "volatility": annual_volatility,
        "sharpe_ratio": sharpe,
        "max_drawdown": expected_max_dd
    }},
    "risk_contributions": {{
        "instrument1": risk_contribution,
        ...
    }},
    "rebalancing_urgency": "immediate|high|medium|low",
    "reasoning": "detailed explanation"
}}"""
    
    def build_rebalancing_analysis_prompt(
        self,
        current_weights: Dict[str, float],
        target_weights: Dict[str, float],
        deviations: Dict[str, float],
        trades_needed: Dict[str, float],
        transaction_costs: float,
        estimated_costs: float,
    ) -> str:
        """Build rebalancing analysis prompt."""
        return f"""Analyze portfolio rebalancing opportunity:

Current Weights: {json.dumps(current_weights, indent=2)}
Target Weights: {json.dumps(target_weights, indent=2)}
Deviations: {json.dumps(deviations, indent=2)}
Trades Needed: {json.dumps(trades_needed, indent=2)}
Transaction Cost Rate: {transaction_costs*100}%
Estimated Total Cost: ${estimated_costs:,.2f}

Provide rebalancing recommendation in JSON format:
{{
    "should_rebalance": true|false,
    "urgency": "immediate|wait|monitor",
    "optimal_trades": {{
        "instrument1": trade_amount,
        ...
    }},
    "expected_improvement": {{
        "tracking_error_reduction": percentage,
        "risk_adjusted_return_improvement": percentage
    }},
    "cost_benefit_analysis": "detailed analysis",
    "alternative_approaches": ["list of alternatives"],
    "reasoning": "detailed explanation"
}}"""
    
    def build_allocation_strategy_prompt(
        self,
        capital: str,
        instruments: List[str],
        risk_profile: str,
        investment_horizon: str,
        market_outlook: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Build allocation strategy prompt."""
        return f"""Design allocation strategy:

Capital: {capital}
Available Instruments: {instruments}
Risk Profile: {risk_profile}
Investment Horizon: {investment_horizon}
{f"Market Outlook: {json.dumps(market_outlook, indent=2)}" if market_outlook else ""}

Provide allocation strategy in JSON format:
{{
    "recommended_allocation": {{
        "instrument1": percentage,
        ...
    }},
    "strategy_type": "strategic|tactical|dynamic",
    "rebalancing_frequency": "monthly|quarterly|threshold-based",
    "risk_management": {{
        "stop_loss_strategy": "description",
        "hedging_approach": "description",
        "volatility_targeting": "description"
    }},
    "expected_performance": {{
        "annual_return": "X-Y%",
        "volatility": "X-Y%",
        "max_drawdown": "X-Y%"
    }},
    "key_risks": ["list of risks"],
    "monitoring_plan": "description"
}}"""


class MarketRegimePrompts(BasePrompts):
    """Prompt templates for market regime detection."""
    
    REGIME_SYSTEM_PROMPT = """You are a market regime expert specializing in:
1. Market state identification
2. Regime transition detection
3. Adaptive strategy recommendations
4. Risk regime analysis
5. Cross-asset regime correlations

Provide precise regime analysis based on quantitative indicators."""
    
    def build_regime_detection_prompt(
        self,
        instrument_id: str,
        indicators: Dict[str, Any],
        recent_bars: List[Dict[str, Any]],
        current_regime: Optional[str] = None,
    ) -> str:
        """Build regime detection prompt."""
        return f"""Detect market regime for {instrument_id}:

Market Indicators:
{json.dumps(indicators, indent=2)}

Recent Price Action:
{json.dumps(recent_bars[-10:], indent=2)}

{f"Current Regime: {current_regime}" if current_regime else "No previous regime detected"}

Provide regime analysis in JSON format:
{{
    "regime": "trending_up|trending_down|ranging|volatile|quiet|transitioning",
    "confidence": 0.0-1.0,
    "characteristics": {{
        "volatility_level": "low|medium|high",
        "trend_strength": 0.0-1.0,
        "momentum": "positive|negative|neutral",
        "volume_profile": "increasing|decreasing|stable"
    }},
    "key_indicators": {{
        "indicator_name": value,
        ...
    }},
    "expected_duration": "hours|days|weeks",
    "trading_implications": "detailed description",
    "optimal_strategies": ["list of suitable strategies"]
}}"""
    
    def build_regime_stability_prompt(
        self,
        current_regime: str,
        stability_metrics: Dict[str, Any],
        recent_transitions: List[Dict[str, Any]],
    ) -> str:
        """Build regime stability analysis prompt."""
        return f"""Analyze regime stability:

Current Regime: {current_regime}
Stability Metrics: {json.dumps(stability_metrics, indent=2)}
Recent Transitions: {json.dumps(recent_transitions, indent=2)}

Provide stability analysis in JSON format:
{{
    "stability_assessment": "stable|unstable|transitioning",
    "transition_probability": {{
        "next_24h": 0.0-1.0,
        "next_week": 0.0-1.0
    }},
    "potential_next_regimes": [
        {{
            "regime": "regime_type",
            "probability": 0.0-1.0,
            "triggers": ["list of potential triggers"]
        }}
    ],
    "warning_signs": ["list of indicators to watch"],
    "recommended_actions": ["list of recommendations"]
}}"""
    
    def build_strategy_adjustment_prompt(
        self,
        regime: str,
        strategy_type: str,
        current_parameters: Dict[str, Any],
        performance_metrics: Optional[Dict[str, float]] = None,
    ) -> str:
        """Build strategy adjustment prompt."""
        return f"""Recommend strategy adjustments for {regime} regime:

Strategy Type: {strategy_type}
Current Parameters: {json.dumps(current_parameters, indent=2)}
{f"Recent Performance: {json.dumps(performance_metrics, indent=2)}" if performance_metrics else ""}

Provide adjustment recommendations in JSON format:
{{
    "parameter_adjustments": {{
        "param_name": {{
            "current": current_value,
            "recommended": new_value,
            "reason": "explanation"
        }},
        ...
    }},
    "position_sizing": {{
        "current_approach": "description",
        "recommended_change": "description",
        "risk_multiplier": 0.1-2.0
    }},
    "entry_exit_rules": {{
        "modifications": ["list of rule changes"],
        "new_filters": ["list of new filters to add"]
    }},
    "expected_improvement": "description",
    "implementation_priority": "immediate|high|medium|low"
}}"""
    
    def build_regime_prediction_prompt(
        self,
        instrument_id: str,
        current_regime: str,
        indicators: Dict[str, Any],
        forecast_horizon: int,
        recent_bars: List[Dict[str, Any]],
    ) -> str:
        """Build regime prediction prompt."""
        return f"""Predict regime changes for {instrument_id}:

Current Regime: {current_regime}
Predictive Indicators: {json.dumps(indicators, indent=2)}
Forecast Horizon: {forecast_horizon} days
Recent Price Data: {json.dumps(recent_bars[-20:], indent=2)}

Provide regime forecast in JSON format:
{{
    "most_likely_scenario": {{
        "regime_sequence": ["current", "next", ...],
        "transition_timing": ["day X", ...],
        "confidence": 0.0-1.0
    }},
    "alternative_scenarios": [
        {{
            "description": "scenario description",
            "probability": 0.0-1.0,
            "regime_sequence": ["regimes"]
        }}
    ],
    "key_catalysts": ["list of potential catalysts"],
    "monitoring_levels": {{
        "price_levels": [level1, level2],
        "indicator_thresholds": {{"indicator": threshold}}
    }},
    "trading_plan": "detailed plan for each scenario"
}}"""


class SentimentAnalysisPrompts(BasePrompts):
    """Prompt templates for sentiment analysis."""
    
    SENTIMENT_SYSTEM_PROMPT = """You are a market sentiment expert specializing in:
1. Multi-source sentiment aggregation
2. Sentiment-price divergence analysis
3. Social media and news sentiment
4. Institutional positioning analysis
5. Sentiment-based trading signals

Provide data-driven sentiment analysis with actionable insights."""
    
    def build_sentiment_analysis_prompt(
        self,
        instrument_id: str,
        text_sources: Dict[str, List[str]],
        market_indicators: Dict[str, Any],
        historical_sentiment: List[Dict[str, Any]],
    ) -> str:
        """Build sentiment analysis prompt."""
        return f"""Analyze market sentiment for {instrument_id}:

Text Sources:
{json.dumps(text_sources, indent=2)}

Market-Based Indicators:
{json.dumps(market_indicators, indent=2)}

Historical Sentiment (last 5):
{json.dumps(historical_sentiment[-5:], indent=2)}

Provide sentiment analysis in JSON format:
{{
    "sentiment": "very_bullish|bullish|neutral|bearish|very_bearish",
    "score": -1.0 to 1.0,  // -1=very bearish, 1=very bullish
    "confidence": 0.0-1.0,
    "sources": {{
        "news": -1.0 to 1.0,
        "social_media": -1.0 to 1.0,
        "analyst_reports": -1.0 to 1.0,
        "market_data": -1.0 to 1.0
    }},
    "key_themes": ["list of dominant themes"],
    "sentiment_drivers": ["list of main drivers"],
    "contrarian_indicators": ["if extreme sentiment detected"],
    "analysis_factors": ["factors considered in analysis"]
}}"""
    
    def build_sentiment_trend_prompt(
        self,
        instrument_id: str,
        historical_sentiment: List[Dict[str, Any]],
        trend_metrics: Dict[str, Any],
    ) -> str:
        """Build sentiment trend analysis prompt."""
        return f"""Analyze sentiment trend for {instrument_id}:

Historical Sentiment Data:
{json.dumps(historical_sentiment, indent=2)}

Trend Metrics:
{json.dumps(trend_metrics, indent=2)}

Provide trend analysis in JSON format:
{{
    "trend": "improving|deteriorating|stable|volatile",
    "trend_strength": 0.0-1.0,
    "momentum": "accelerating|decelerating|steady",
    "turning_points": [
        {{
            "type": "peak|trough",
            "timestamp": "ISO timestamp",
            "significance": "high|medium|low"
        }}
    ],
    "forecast": {{
        "next_24h": "direction",
        "next_week": "direction",
        "confidence": 0.0-1.0
    }},
    "regime_alignment": "sentiment aligns/diverges with price",
    "trading_implications": "detailed implications"
}}"""
    
    def build_sentiment_divergence_prompt(
        self,
        instrument_id: str,
        price_trend: str,
        sentiment_trend: str,
        correlation: float,
        aligned_data: List[Dict[str, Any]],
    ) -> str:
        """Build sentiment divergence analysis prompt."""
        return f"""Analyze price-sentiment divergence for {instrument_id}:

Price Trend: {price_trend}
Sentiment Trend: {sentiment_trend}
Correlation: {correlation}

Aligned Price & Sentiment Data (recent 10):
{json.dumps(aligned_data[-10:], indent=2)}

Provide divergence analysis in JSON format:
{{
    "divergence_detected": true|false,
    "divergence_type": "bullish|bearish|none",
    "divergence_strength": 0.0-1.0,
    "historical_outcome_probability": {{
        "price_follows_sentiment": 0.0-1.0,
        "sentiment_follows_price": 0.0-1.0,
        "convergence_timeframe": "days"
    }},
    "trading_signal": {{
        "action": "buy|sell|wait",
        "confidence": 0.0-1.0,
        "reasoning": "explanation"
    }},
    "risk_factors": ["list of risks"],
    "monitoring_plan": "what to watch for"
}}"""
    
    def build_sentiment_signal_prompt(
        self,
        instrument_id: str,
        current_sentiment: Dict[str, Any],
        sentiment_trend: Dict[str, Any],
        current_price: str,
        risk_tolerance: str,
    ) -> str:
        """Build sentiment-based signal generation prompt."""
        return f"""Generate trading signals based on sentiment for {instrument_id}:

Current Sentiment:
{json.dumps(current_sentiment, indent=2)}

Sentiment Trend:
{json.dumps(sentiment_trend, indent=2)}

Current Price: {current_price}
Risk Tolerance: {risk_tolerance}

Provide trading signals in JSON format:
{{
    "primary_signal": {{
        "action": "long|short|neutral",
        "strength": 0.0-1.0,
        "entry_timing": "immediate|wait_for_confirmation|scale_in",
        "position_size": "full|half|quarter",
        "reasoning": "detailed explanation"
    }},
    "risk_management": {{
        "stop_loss_distance": "X%",
        "take_profit_targets": ["TP1", "TP2"],
        "position_management": "instructions"
    }},
    "confirmation_criteria": ["what to look for"],
    "invalidation_scenarios": ["what would invalidate signal"],
    "time_horizon": "hours|days|weeks",
    "confidence_factors": ["factors supporting confidence level"]
}}"""


class AnomalyDetectionPrompts(BasePrompts):
    """Prompt templates for anomaly detection."""
    
    ANOMALY_SYSTEM_PROMPT = """You are a market anomaly detection expert specializing in:
1. Price and volume anomalies
2. Market microstructure irregularities  
3. Cross-market anomaly propagation
4. Flash events and fat finger detection
5. Market manipulation patterns

Provide precise anomaly identification with severity assessment."""
    
    def build_anomaly_detection_prompt(
        self,
        instrument_id: str,
        indicators: Dict[str, Any],
        statistical_anomalies: List[Dict[str, Any]],
        recent_bars: List[Dict[str, Any]],
        trades_summary: Optional[Dict[str, Any]] = None,
        quotes_summary: Optional[Dict[str, Any]] = None,
        reference_data: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Build anomaly detection prompt."""
        return f"""Detect market anomalies for {instrument_id}:

Market Indicators:
{json.dumps(indicators, indent=2)}

Statistical Anomalies Detected:
{json.dumps(statistical_anomalies, indent=2)}

Recent Price Bars:
{json.dumps(recent_bars[-10:], indent=2)}

{f"Trades Summary: {json.dumps(trades_summary, indent=2)}" if trades_summary else ""}
{f"Quotes Summary: {json.dumps(quotes_summary, indent=2)}" if quotes_summary else ""}
{f"Reference Data: {json.dumps(reference_data, indent=2)}" if reference_data else ""}

Provide anomaly analysis in JSON format:
{{
    "anomalies": [
        {{
            "type": "price_spike|volume_surge|spread_widening|liquidity_drop|flash_crash|fat_finger|manipulation",
            "severity": "low|medium|high|critical",
            "confidence": 0.0-1.0,
            "start_time": "ISO timestamp",
            "end_time": "ISO timestamp or null if ongoing",
            "description": "detailed description",
            "metrics": {{
                "key_metric": value,
                ...
            }},
            "potential_causes": ["list of possible causes"],
            "affected_instruments": ["list if cross-market impact"]
        }}
    ],
    "market_quality_score": 0.0-1.0,  // 1=normal, 0=severely anomalous
    "recommended_actions": ["list of recommended actions"],
    "follow_up_monitoring": ["what to monitor"]
}}"""
    
    def build_anomaly_impact_prompt(
        self,
        anomaly: Dict[str, Any],
        market_data: Dict[str, Any],
        positions: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """Build anomaly impact analysis prompt."""
        return f"""Analyze impact of detected anomaly:

Anomaly Details:
{json.dumps(anomaly, indent=2)}

Current Market Data:
{json.dumps(market_data, indent=2)}

{f"Active Positions: {json.dumps(positions, indent=2)}" if positions else "No active positions"}

Provide impact analysis in JSON format:
{{
    "immediate_impact": {{
        "price_impact": "X%",
        "liquidity_impact": "description",
        "spread_impact": "X bps",
        "trading_halt_risk": "low|medium|high"
    }},
    "position_impact": {{
        "unrealized_pnl_impact": "amount or percentage",
        "execution_risk": "description",
        "recommended_position_adjustments": ["list of adjustments"]
    }},
    "market_contagion_risk": {{
        "probability": 0.0-1.0,
        "potentially_affected_markets": ["list of markets"],
        "estimated_duration": "minutes|hours|days"
    }},
    "recovery_scenarios": [
        {{
            "scenario": "description",
            "probability": 0.0-1.0,
            "timeline": "expected timeline"
        }}
    ],
    "action_priority": "immediate|high|medium|monitor"
}}"""
    
    def build_anomaly_response_prompt(
        self,
        anomaly: Dict[str, Any],
        current_strategy: str,
        risk_limits: Dict[str, Any],
        market_conditions: Optional[Dict[str, Any]] = None,
        similar_anomalies: List[Dict[str, Any]] = None,
    ) -> str:
        """Build anomaly response recommendation prompt."""
        return f"""Recommend response to market anomaly:

Anomaly:
{json.dumps(anomaly, indent=2)}

Current Strategy: {current_strategy}
Risk Limits: {json.dumps(risk_limits, indent=2)}
{f"Market Conditions: {json.dumps(market_conditions, indent=2)}" if market_conditions else ""}

{f"Similar Historical Anomalies: {json.dumps(similar_anomalies, indent=2)}" if similar_anomalies else ""}

Provide response recommendations in JSON format:
{{
    "immediate_actions": [
        {{
            "action": "description",
            "priority": "critical|high|medium",
            "reasoning": "explanation"
        }}
    ],
    "position_management": {{
        "close_positions": true|false,
        "reduce_exposure": "percentage",
        "hedging_strategy": "description",
        "stop_loss_adjustments": "tighten|maintain|widen"
    }},
    "trading_adjustments": {{
        "pause_trading": true|false,
        "parameter_overrides": {{"param": "value"}},
        "strategy_modifications": ["list of modifications"]
    }},
    "monitoring_plan": {{
        "key_metrics": ["metrics to watch"],
        "alert_thresholds": {{"metric": "threshold"}},
        "review_frequency": "minutes"
    }},
    "contingency_plans": [
        {{
            "trigger": "condition",
            "action": "response"
        }}
    ]
}}"""
    
    def build_anomaly_monitoring_prompt(
        self,
        anomaly: Dict[str, Any],
        current_data: Dict[str, Any],
        duration_minutes: float,
    ) -> str:
        """Build anomaly monitoring prompt."""
        return f"""Monitor anomaly resolution:

Active Anomaly:
{json.dumps(anomaly, indent=2)}

Current Market Data:
{json.dumps(current_data, indent=2)}

Anomaly Duration: {duration_minutes:.1f} minutes

Provide monitoring update in JSON format:
{{
    "status": "active|resolving|resolved|escalating",
    "is_resolved": true|false,
    "severity_update": "unchanged|increased|decreased",
    "resolution_progress": 0.0-1.0,
    "new_developments": ["list of new observations"],
    "updated_metrics": {{
        "metric": "current_value",
        ...
    }},
    "expected_resolution_time": "minutes remaining or 'unknown'",
    "recommended_actions": ["based on current status"],
    "escalation_risk": "low|medium|high"
}}"""

# Re-export for compatibility
__all__ = [
    "BasePrompts",
    "MarketAnalysisPrompts",
    "RiskAnalysisPrompts",
    "StrategyAuditPrompts",
    "PortfolioOptimizationPrompts",
    "MarketRegimePrompts",
    "SentimentAnalysisPrompts",
    "AnomalyDetectionPrompts",
]