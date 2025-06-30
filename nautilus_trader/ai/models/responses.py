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
Pydantic models for AI response validation.
"""

from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import List, Optional, Any, Dict

from pydantic import BaseModel, Field, validator


class Direction(str, Enum):
    """Market direction."""
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


class PatternType(str, Enum):
    """Technical pattern types."""
    CONTINUATION = "continuation"
    REVERSAL = "reversal"


class Timeframe(str, Enum):
    """Trading timeframes."""
    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    H1 = "1h"
    H4 = "4h"
    D1 = "1d"


class VolatilityRegime(str, Enum):
    """Volatility regimes."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"


class TrendAnalysisResponse(BaseModel):
    """Validated response for trend analysis."""
    
    direction: Direction
    strength: float = Field(ge=0.0, le=1.0, description="Trend strength from 0 to 1")
    confidence: float = Field(ge=0.0, le=1.0, description="Analysis confidence from 0 to 1")
    support_levels: List[float] = Field(min_items=0, max_items=5)
    resistance_levels: List[float] = Field(min_items=0, max_items=5)
    reasoning: str = Field(min_length=10, max_length=1000)
    
    @validator('support_levels', 'resistance_levels')
    def validate_price_levels(cls, v):
        """Ensure price levels are positive and sorted."""
        if v:
            sorted_levels = sorted([level for level in v if level > 0])
            return sorted_levels
        return v
    
    class Config:
        use_enum_values = True
        

class SignalGenerationResponse(BaseModel):
    """Validated response for signal generation."""
    
    generate_signal: bool
    side: Optional[str] = Field(None, regex="^(buy|sell)$")
    confidence: float = Field(ge=0.0, le=1.0)
    entry_price: Optional[float] = Field(None, gt=0)
    stop_loss: Optional[float] = Field(None, gt=0)
    take_profit: Optional[float] = Field(None, gt=0)
    reasoning: str = Field(min_length=10, max_length=1000)
    timeframe: Optional[Timeframe] = None
    
    @validator('side', 'entry_price', 'stop_loss', 'take_profit')
    def validate_signal_fields(cls, v, values):
        """Ensure signal fields are present when signal is generated."""
        if values.get('generate_signal') and v is None:
            raise ValueError(f"Field is required when generate_signal is True")
        return v
    
    @validator('stop_loss')
    def validate_stop_loss(cls, v, values):
        """Ensure stop loss is logical based on side."""
        if v and values.get('entry_price') and values.get('side'):
            if values['side'] == 'buy' and v >= values['entry_price']:
                raise ValueError("Stop loss must be below entry price for buy orders")
            elif values['side'] == 'sell' and v <= values['entry_price']:
                raise ValueError("Stop loss must be above entry price for sell orders")
        return v
    

class PatternRecognitionResponse(BaseModel):
    """Validated response for pattern recognition."""
    
    class Pattern(BaseModel):
        """Individual pattern."""
        name: str = Field(min_length=3, max_length=50)
        type: PatternType
        confidence: float = Field(ge=0.0, le=1.0)
        target_price: Optional[float] = Field(None, gt=0)
        description: str = Field(min_length=10, max_length=500)
    
    patterns: List[Pattern] = Field(max_items=10)
    
    @validator('patterns')
    def filter_low_confidence(cls, v):
        """Filter out patterns with confidence below 0.6."""
        return [p for p in v if p.confidence >= 0.6]
    

class VolatilityAnalysisResponse(BaseModel):
    """Validated response for volatility analysis."""
    
    predicted_volatility: str = Field(min_length=1, max_length=100)
    regime: VolatilityRegime
    trend: str = Field(regex="^(increasing|stable|decreasing)$")
    risk_assessment: str = Field(min_length=10, max_length=500)
    trading_implications: str = Field(min_length=10, max_length=500)
    

class RiskAssessmentResponse(BaseModel):
    """Validated response for risk assessment."""
    
    class RiskFactor(BaseModel):
        """Individual risk factor."""
        factor: str = Field(min_length=3, max_length=100)
        severity: str = Field(regex="^(low|medium|high)$")
        description: str = Field(min_length=10, max_length=500)
    
    risk_score: float = Field(ge=0.0, le=1.0)
    risk_factors: List[RiskFactor] = Field(max_items=20)
    var_95: Optional[str] = None
    max_drawdown_expected: Optional[str] = None
    recommendations: List[str] = Field(min_items=1, max_items=10)
    stop_loss_adjustment: Optional[float] = Field(None, gt=0)
    

class PortfolioRiskResponse(BaseModel):
    """Validated response for portfolio risk analysis."""
    
    portfolio_risk_score: float = Field(ge=0.0, le=1.0)
    diversification_score: float = Field(ge=0.0, le=1.0)
    concentration_risks: List[str] = Field(max_items=10)
    correlation_risks: List[str] = Field(max_items=10)
    stress_test_results: Dict[str, str] = Field(max_items=10)
    recommendations: List[str] = Field(min_items=1, max_items=10)
    optimal_hedge_suggestions: List[str] = Field(max_items=5)
    

class AIResponseAdapter:
    """
    Adapter to parse and validate AI responses.
    
    This provides a clean interface between raw AI responses
    and the application, ensuring type safety and validation.
    """
    
    @staticmethod
    def parse_trend_analysis(raw_response: str) -> TrendAnalysisResponse:
        """Parse and validate trend analysis response."""
        return TrendAnalysisResponse.parse_raw(raw_response)
    
    @staticmethod
    def parse_signal_generation(raw_response: str) -> SignalGenerationResponse:
        """Parse and validate signal generation response."""
        return SignalGenerationResponse.parse_raw(raw_response)
    
    @staticmethod
    def parse_pattern_recognition(raw_response: str) -> PatternRecognitionResponse:
        """Parse and validate pattern recognition response."""
        return PatternRecognitionResponse.parse_raw(raw_response)
    
    @staticmethod
    def parse_volatility_analysis(raw_response: str) -> VolatilityAnalysisResponse:
        """Parse and validate volatility analysis response."""
        return VolatilityAnalysisResponse.parse_raw(raw_response)
    
    @staticmethod
    def parse_risk_assessment(raw_response: str) -> RiskAssessmentResponse:
        """Parse and validate risk assessment response."""
        return RiskAssessmentResponse.parse_raw(raw_response)
    
    @staticmethod
    def parse_portfolio_risk(raw_response: str) -> PortfolioRiskResponse:
        """Parse and validate portfolio risk response."""
        return PortfolioRiskResponse.parse_raw(raw_response)
    
    @staticmethod
    def safe_parse(response_type: str, raw_response: str) -> Optional[BaseModel]:
        """
        Safely parse response with fallback to None.
        
        Parameters
        ----------
        response_type : str
            Type of response to parse
        raw_response : str
            Raw JSON response from AI
            
        Returns
        -------
        BaseModel or None
            Parsed response or None if parsing fails
        """
        parsers = {
            "trend_analysis": AIResponseAdapter.parse_trend_analysis,
            "signal_generation": AIResponseAdapter.parse_signal_generation,
            "pattern_recognition": AIResponseAdapter.parse_pattern_recognition,
            "volatility_analysis": AIResponseAdapter.parse_volatility_analysis,
            "risk_assessment": AIResponseAdapter.parse_risk_assessment,
            "portfolio_risk": AIResponseAdapter.parse_portfolio_risk,
        }
        
        parser = parsers.get(response_type)
        if not parser:
            return None
            
        try:
            return parser(raw_response)
        except Exception as e:
            # Log error and return None
            print(f"Failed to parse {response_type}: {e}")
            return None