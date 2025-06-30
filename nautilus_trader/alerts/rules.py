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
Alert rules for automated monitoring and alerting.
"""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Union
from datetime import datetime, timedelta, timezone
from enum import Enum
import re

from nautilus_trader.alerts.base import Alert, AlertLevel, AlertManager
from nautilus_trader.common.component import Logger


class RuleOperator(str, Enum):
    """Comparison operators for rules."""
    EQUALS = "=="
    NOT_EQUALS = "!="
    GREATER_THAN = ">"
    GREATER_THAN_OR_EQUAL = ">="
    LESS_THAN = "<"
    LESS_THAN_OR_EQUAL = "<="
    CONTAINS = "contains"
    NOT_CONTAINS = "not_contains"
    MATCHES = "matches"  # Regex
    IN = "in"
    NOT_IN = "not_in"


class AlertRule(ABC):
    """
    Abstract base class for alert rules.
    
    Alert rules evaluate conditions and trigger alerts when met.
    """
    
    def __init__(
        self,
        name: str,
        level: AlertLevel = AlertLevel.INFO,
        enabled: bool = True,
        cooldown_seconds: int = 0,
    ) -> None:
        """
        Initialize alert rule.
        
        Parameters
        ----------
        name : str
            Rule name
        level : AlertLevel
            Alert level when triggered
        enabled : bool
            Whether rule is enabled
        cooldown_seconds : int
            Minimum seconds between alerts
        """
        self.name = name
        self.level = level
        self.enabled = enabled
        self.cooldown_seconds = cooldown_seconds
        self._last_triggered: Optional[datetime] = None
        self._logger = Logger(f"{self.__class__.__name__}[{name}]")
        
    @abstractmethod
    def evaluate(self, context: Dict[str, Any]) -> Optional[str]:
        """
        Evaluate the rule.
        
        Parameters
        ----------
        context : dict
            Evaluation context
            
        Returns
        -------
        str or None
            Alert message if triggered, None otherwise
        """
        pass
        
    def check(self, context: Dict[str, Any]) -> Optional[Alert]:
        """
        Check rule and create alert if triggered.
        
        Parameters
        ----------
        context : dict
            Evaluation context
            
        Returns
        -------
        Alert or None
            Alert if rule triggered
        """
        if not self.enabled:
            return None
            
        # Check cooldown
        if self._last_triggered and self.cooldown_seconds > 0:
            elapsed = (datetime.now(timezone.utc) - self._last_triggered).total_seconds()
            if elapsed < self.cooldown_seconds:
                return None
                
        # Evaluate rule
        message = self.evaluate(context)
        
        if message:
            self._last_triggered = datetime.now(timezone.utc)
            
            return Alert.create(
                level=self.level,
                title=f"Rule Triggered: {self.name}",
                message=message,
                source=f"rule:{self.name}",
                rule_name=self.name,
                context=context,
            )
            
        return None


class ThresholdRule(AlertRule):
    """
    Rule that triggers when a value crosses a threshold.
    
    Supports various comparison operators and optional hysteresis.
    """
    
    def __init__(
        self,
        name: str,
        field: str,
        threshold: Union[float, int],
        operator: Union[RuleOperator, str] = RuleOperator.GREATER_THAN,
        hysteresis: Optional[float] = None,
        level: AlertLevel = AlertLevel.WARNING,
        enabled: bool = True,
        cooldown_seconds: int = 60,
    ) -> None:
        """
        Initialize threshold rule.
        
        Parameters
        ----------
        name : str
            Rule name
        field : str
            Field to monitor
        threshold : float or int
            Threshold value
        operator : RuleOperator
            Comparison operator
        hysteresis : float, optional
            Hysteresis to prevent flapping
        level : AlertLevel
            Alert level
        enabled : bool
            Whether enabled
        cooldown_seconds : int
            Cooldown period
        """
        super().__init__(name, level, enabled, cooldown_seconds)
        self.field = field
        self.threshold = threshold
        self.operator = RuleOperator(operator) if isinstance(operator, str) else operator
        self.hysteresis = hysteresis
        self._triggered = False
        
    def evaluate(self, context: Dict[str, Any]) -> Optional[str]:
        """Evaluate threshold rule."""
        # Get value from context
        value = self._get_field_value(context, self.field)
        if value is None:
            return None
            
        # Apply hysteresis if configured
        effective_threshold = self._get_effective_threshold()
        
        # Compare value
        triggered = self._compare(value, effective_threshold)
        
        if triggered and not self._triggered:
            self._triggered = True
            return (
                f"{self.field} is {value} "
                f"(threshold: {self.operator.value} {self.threshold})"
            )
        elif not triggered:
            self._triggered = False
            
        return None
        
    def _get_effective_threshold(self) -> Union[float, int]:
        """Get threshold with hysteresis applied."""
        if not self.hysteresis or not self._triggered:
            return self.threshold
            
        # Apply hysteresis based on operator
        if self.operator in [RuleOperator.GREATER_THAN, RuleOperator.GREATER_THAN_OR_EQUAL]:
            return self.threshold - self.hysteresis
        elif self.operator in [RuleOperator.LESS_THAN, RuleOperator.LESS_THAN_OR_EQUAL]:
            return self.threshold + self.hysteresis
        else:
            return self.threshold
            
    def _compare(self, value: Any, threshold: Any) -> bool:
        """Compare value against threshold."""
        try:
            if self.operator == RuleOperator.EQUALS:
                return value == threshold
            elif self.operator == RuleOperator.NOT_EQUALS:
                return value != threshold
            elif self.operator == RuleOperator.GREATER_THAN:
                return value > threshold
            elif self.operator == RuleOperator.GREATER_THAN_OR_EQUAL:
                return value >= threshold
            elif self.operator == RuleOperator.LESS_THAN:
                return value < threshold
            elif self.operator == RuleOperator.LESS_THAN_OR_EQUAL:
                return value <= threshold
            elif self.operator == RuleOperator.CONTAINS:
                return str(threshold) in str(value)
            elif self.operator == RuleOperator.NOT_CONTAINS:
                return str(threshold) not in str(value)
            elif self.operator == RuleOperator.MATCHES:
                return bool(re.match(str(threshold), str(value)))
            elif self.operator == RuleOperator.IN:
                return value in threshold
            elif self.operator == RuleOperator.NOT_IN:
                return value not in threshold
            else:
                return False
        except Exception as e:
            self._logger.error(f"Comparison failed: {e}")
            return False
            
    def _get_field_value(self, context: Dict[str, Any], field: str) -> Any:
        """Get field value from context (supports nested fields)."""
        parts = field.split(".")
        value = context
        
        for part in parts:
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                return None
                
        return value


class ChangeRule(AlertRule):
    """
    Rule that triggers on value changes.
    
    Can detect increases, decreases, or any change.
    """
    
    def __init__(
        self,
        name: str,
        field: str,
        change_type: str = "any",  # "any", "increase", "decrease"
        min_change: Optional[float] = None,
        percentage: bool = False,
        level: AlertLevel = AlertLevel.INFO,
        enabled: bool = True,
        cooldown_seconds: int = 60,
    ) -> None:
        """
        Initialize change rule.
        
        Parameters
        ----------
        name : str
            Rule name
        field : str
            Field to monitor
        change_type : str
            Type of change to detect
        min_change : float, optional
            Minimum change amount
        percentage : bool
            Whether min_change is percentage
        level : AlertLevel
            Alert level
        enabled : bool
            Whether enabled
        cooldown_seconds : int
            Cooldown period
        """
        super().__init__(name, level, enabled, cooldown_seconds)
        self.field = field
        self.change_type = change_type
        self.min_change = min_change
        self.percentage = percentage
        self._last_value: Optional[Any] = None
        
    def evaluate(self, context: Dict[str, Any]) -> Optional[str]:
        """Evaluate change rule."""
        # Get current value
        current_value = self._get_field_value(context, self.field)
        if current_value is None:
            return None
            
        # First time - just store value
        if self._last_value is None:
            self._last_value = current_value
            return None
            
        # Calculate change
        try:
            change = current_value - self._last_value
            
            if self.percentage and self._last_value != 0:
                change_pct = (change / self._last_value) * 100
                change_amount = abs(change_pct)
                change_str = f"{change_pct:+.2f}%"
            else:
                change_amount = abs(change)
                change_str = f"{change:+.2f}"
                
        except (TypeError, ValueError):
            # Non-numeric change
            if current_value != self._last_value:
                self._last_value = current_value
                if self.change_type == "any":
                    return f"{self.field} changed from {self._last_value} to {current_value}"
            return None
            
        # Check change type
        triggered = False
        
        if self.change_type == "any" and change != 0:
            triggered = True
        elif self.change_type == "increase" and change > 0:
            triggered = True
        elif self.change_type == "decrease" and change < 0:
            triggered = True
            
        # Check minimum change
        if triggered and self.min_change is not None:
            triggered = change_amount >= self.min_change
            
        # Update last value
        self._last_value = current_value
        
        if triggered:
            return (
                f"{self.field} changed by {change_str} "
                f"(from {self._last_value} to {current_value})"
            )
            
        return None
        
    def _get_field_value(self, context: Dict[str, Any], field: str) -> Any:
        """Get field value from context."""
        parts = field.split(".")
        value = context
        
        for part in parts:
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                return None
                
        return value


class PatternRule(AlertRule):
    """
    Rule that triggers on specific patterns or conditions.
    
    Uses a custom evaluation function.
    """
    
    def __init__(
        self,
        name: str,
        pattern_func: Callable[[Dict[str, Any]], Optional[str]],
        level: AlertLevel = AlertLevel.INFO,
        enabled: bool = True,
        cooldown_seconds: int = 60,
    ) -> None:
        """
        Initialize pattern rule.
        
        Parameters
        ----------
        name : str
            Rule name
        pattern_func : callable
            Function that evaluates pattern
        level : AlertLevel
            Alert level
        enabled : bool
            Whether enabled
        cooldown_seconds : int
            Cooldown period
        """
        super().__init__(name, level, enabled, cooldown_seconds)
        self.pattern_func = pattern_func
        
    def evaluate(self, context: Dict[str, Any]) -> Optional[str]:
        """Evaluate pattern rule."""
        try:
            return self.pattern_func(context)
        except Exception as e:
            self._logger.error(f"Pattern evaluation failed: {e}")
            return None


class CompositeRule(AlertRule):
    """
    Rule that combines multiple rules with logical operators.
    
    Supports AND, OR, and NOT operations.
    """
    
    def __init__(
        self,
        name: str,
        rules: List[AlertRule],
        operator: str = "AND",  # "AND", "OR"
        level: Optional[AlertLevel] = None,
        enabled: bool = True,
        cooldown_seconds: int = 60,
    ) -> None:
        """
        Initialize composite rule.
        
        Parameters
        ----------
        name : str
            Rule name
        rules : list[AlertRule]
            Rules to combine
        operator : str
            Logical operator
        level : AlertLevel, optional
            Override alert level
        enabled : bool
            Whether enabled
        cooldown_seconds : int
            Cooldown period
        """
        # Use highest level from child rules if not specified
        if level is None:
            levels = [AlertLevel.DEBUG, AlertLevel.INFO, AlertLevel.WARNING,
                     AlertLevel.ERROR, AlertLevel.CRITICAL]
            rule_levels = [r.level for r in rules]
            level = max(rule_levels, key=lambda x: levels.index(x))
            
        super().__init__(name, level, enabled, cooldown_seconds)
        self.rules = rules
        self.operator = operator.upper()
        
    def evaluate(self, context: Dict[str, Any]) -> Optional[str]:
        """Evaluate composite rule."""
        results = []
        
        for rule in self.rules:
            if not rule.enabled:
                continue
                
            # Temporarily disable cooldown for child rules
            original_last_triggered = rule._last_triggered
            rule._last_triggered = None
            
            message = rule.evaluate(context)
            
            # Restore cooldown
            rule._last_triggered = original_last_triggered
            
            results.append((rule, message))
            
        # Apply logical operator
        if self.operator == "AND":
            # All rules must trigger
            triggered_rules = [r for r, m in results if m is not None]
            if len(triggered_rules) == len(self.rules):
                messages = [f"{r.name}: {m}" for r, m in results if m]
                return " AND ".join(messages)
                
        elif self.operator == "OR":
            # Any rule must trigger
            triggered = [(r, m) for r, m in results if m is not None]
            if triggered:
                messages = [f"{r.name}: {m}" for r, m in triggered]
                return " OR ".join(messages)
                
        return None


class RuleSet:
    """
    Collection of alert rules that can be evaluated together.
    """
    
    def __init__(
        self,
        name: str,
        alert_manager: AlertManager,
    ) -> None:
        """
        Initialize rule set.
        
        Parameters
        ----------
        name : str
            Rule set name
        alert_manager : AlertManager
            Alert manager for sending alerts
        """
        self.name = name
        self._alert_manager = alert_manager
        self._rules: Dict[str, AlertRule] = {}
        self._logger = Logger(f"{self.__class__.__name__}[{name}]")
        
    def add_rule(self, rule: AlertRule) -> None:
        """Add a rule to the set."""
        self._rules[rule.name] = rule
        self._logger.info(f"Added rule: {rule.name}")
        
    def remove_rule(self, name: str) -> None:
        """Remove a rule from the set."""
        if name in self._rules:
            del self._rules[name]
            self._logger.info(f"Removed rule: {name}")
            
    def enable_rule(self, name: str) -> None:
        """Enable a rule."""
        if name in self._rules:
            self._rules[name].enabled = True
            
    def disable_rule(self, name: str) -> None:
        """Disable a rule."""
        if name in self._rules:
            self._rules[name].enabled = False
            
    async def evaluate(self, context: Dict[str, Any]) -> int:
        """
        Evaluate all rules and send alerts.
        
        Parameters
        ----------
        context : dict
            Evaluation context
            
        Returns
        -------
        int
            Number of alerts triggered
        """
        alerts_sent = 0
        
        for rule in self._rules.values():
            if not rule.enabled:
                continue
                
            try:
                alert = rule.check(context)
                if alert:
                    await self._alert_manager.send_alert(
                        level=alert.level,
                        title=alert.title,
                        message=alert.message,
                        source=alert.source,
                        **alert.metadata,
                    )
                    alerts_sent += 1
                    
            except Exception as e:
                self._logger.error(f"Rule evaluation failed for {rule.name}: {e}")
                
        return alerts_sent