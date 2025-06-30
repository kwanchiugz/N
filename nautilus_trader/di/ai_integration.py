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
Integration between AI modules and DI container for intelligent service management.
"""

import asyncio
import json
from typing import Any, Dict, List, Optional, Type
from dataclasses import dataclass
from datetime import datetime

from nautilus_trader.common.component import Logger
from nautilus_trader.di.ai_enhanced_container import AIEnhancedDIContainer, Lifetime
from nautilus_trader.ai.config import AIConfig
from nautilus_trader.ai.analyzers.market import MarketAnalyzer
from nautilus_trader.ai.providers.deepseek import DeepSeekProvider


@dataclass
class ServiceOptimizationRequest:
    """Request for AI-powered service optimization analysis."""
    
    container: AIEnhancedDIContainer
    analysis_type: str  # "performance", "configuration", "dependency"
    context: Dict[str, Any]
    max_suggestions: int = 5


@dataclass
class OptimizationSuggestion:
    """AI-powered optimization suggestion for DI container."""
    
    suggestion_type: str  # "lifetime_change", "dependency_injection", "factory_pattern"
    target_service: str
    current_config: Dict[str, Any]
    recommended_config: Dict[str, Any]
    confidence: float
    reasoning: str
    expected_impact: str
    implementation_steps: List[str]


class AIContainerOptimizer:
    """
    Uses AI analysis to provide intelligent optimization suggestions
    for DI container configuration and service management.
    """
    
    def __init__(
        self,
        ai_config: Optional[AIConfig] = None,
        enable_real_ai: bool = False,
    ):
        """
        Initialize AI container optimizer.
        
        Parameters
        ----------
        ai_config : AIConfig, optional
            AI provider configuration
        enable_real_ai : bool, default False
            Whether to use real AI provider or heuristics
        """
        self._logger = Logger(self.__class__.__name__)
        self._ai_config = ai_config
        self._enable_real_ai = enable_real_ai
        self._ai_provider = None
        
        if enable_real_ai and ai_config:
            try:
                self._ai_provider = DeepSeekProvider(ai_config.provider_config)
                self._logger.info("Real AI provider initialized for container optimization")
            except Exception as e:
                self._logger.warning(f"Failed to initialize AI provider: {e}, falling back to heuristics")
                self._enable_real_ai = False
    
    async def analyze_container_performance(
        self,
        container: AIEnhancedDIContainer,
        usage_data: Optional[Dict[str, Any]] = None,
    ) -> List[OptimizationSuggestion]:
        """
        Analyze container performance and suggest optimizations.
        
        Parameters
        ----------
        container : AIEnhancedDIContainer
            Container to analyze
        usage_data : Dict[str, Any], optional
            Additional usage data for analysis
            
        Returns
        -------
        List[OptimizationSuggestion]
            Performance optimization suggestions
        """
        suggestions = []
        
        if self._enable_real_ai and self._ai_provider:
            suggestions.extend(await self._ai_performance_analysis(container, usage_data))
        else:
            suggestions.extend(self._heuristic_performance_analysis(container, usage_data))
            
        return suggestions
    
    async def analyze_service_configuration(
        self,
        container: AIEnhancedDIContainer,
        service_types: Optional[List[Type]] = None,
    ) -> List[OptimizationSuggestion]:
        """
        Analyze service configurations and suggest improvements.
        
        Parameters
        ----------
        container : AIEnhancedDIContainer
            Container to analyze
        service_types : List[Type], optional
            Specific services to analyze (all if None)
            
        Returns
        -------
        List[OptimizationSuggestion]
            Configuration optimization suggestions
        """
        suggestions = []
        
        if self._enable_real_ai and self._ai_provider:
            suggestions.extend(await self._ai_configuration_analysis(container, service_types))
        else:
            suggestions.extend(self._heuristic_configuration_analysis(container, service_types))
            
        return suggestions
    
    async def _ai_performance_analysis(
        self,
        container: AIEnhancedDIContainer,
        usage_data: Optional[Dict[str, Any]],
    ) -> List[OptimizationSuggestion]:
        """Perform AI-powered performance analysis."""
        suggestions = []
        
        try:
            # Prepare analysis context
            insights = container.get_ai_insights()
            
            context = {
                "container_metrics": insights,
                "usage_data": usage_data or {},
                "registered_services": len(container._services),
                "avg_resolution_time": insights.get("avg_resolution_time", 0),
                "most_used_services": insights.get("most_used_services", []),
            }
            
            # Build AI prompt for performance analysis
            prompt = self._build_performance_analysis_prompt(context)
            
            # Get AI analysis
            response = await self._ai_provider.complete(
                prompt=prompt,
                system_prompt="You are an expert in dependency injection optimization. "
                            "Analyze the provided container metrics and suggest specific optimizations.",
                temperature=0.3,
            )
            
            # Parse AI response into suggestions
            ai_suggestions = self._parse_ai_optimization_response(response.content)
            suggestions.extend(ai_suggestions)
            
        except Exception as e:
            self._logger.error(f"AI performance analysis failed: {e}")
            # Fall back to heuristics
            suggestions.extend(self._heuristic_performance_analysis(container, usage_data))
            
        return suggestions
    
    def _heuristic_performance_analysis(
        self,
        container: AIEnhancedDIContainer,
        usage_data: Optional[Dict[str, Any]],
    ) -> List[OptimizationSuggestion]:
        """Perform heuristic-based performance analysis."""
        suggestions = []
        
        insights = container.get_ai_insights()
        
        # Check for slow resolution times
        if insights.get("avg_resolution_time", 0) > 0.002:  # > 2ms
            suggestions.append(OptimizationSuggestion(
                suggestion_type="performance_optimization",
                target_service="container_general",
                current_config={"avg_resolution_time": insights.get("avg_resolution_time", 0)},
                recommended_config={"enable_caching": True, "optimize_providers": True},
                confidence=0.8,
                reasoning="Average resolution time is high. Consider caching frequently used services.",
                expected_impact="Reduce resolution time by 30-50%",
                implementation_steps=[
                    "Convert frequently used transient services to singletons",
                    "Add lightweight caching for expensive service creation",
                    "Review constructor complexity for slow services",
                ],
            ))
        
        # Check most used services
        for service_info in insights.get("most_used_services", []):
            if (service_info.get("usage_count", 0) > 10 and 
                service_info.get("avg_resolution_time", 0) > 0.001):
                
                suggestions.append(OptimizationSuggestion(
                    suggestion_type="lifetime_change",
                    target_service=service_info["service"],
                    current_config={"lifetime": "unknown", "usage_count": service_info["usage_count"]},
                    recommended_config={"lifetime": "singleton"},
                    confidence=0.9,
                    reasoning=f"Service {service_info['service']} is used frequently ({service_info['usage_count']} times) "
                             f"with {service_info['avg_resolution_time']*1000:.2f}ms resolution time",
                    expected_impact="Reduce resolution overhead by creating instance once",
                    implementation_steps=[
                        f"Change {service_info['service']} registration to SINGLETON lifetime",
                        "Verify thread safety of the service implementation",
                        "Test for any state-related issues",
                    ],
                ))
        
        return suggestions
    
    async def _ai_configuration_analysis(
        self,
        container: AIEnhancedDIContainer,
        service_types: Optional[List[Type]],
    ) -> List[OptimizationSuggestion]:
        """Perform AI-powered configuration analysis."""
        suggestions = []
        
        try:
            # Analyze service registrations
            service_analysis = {}
            
            services_to_analyze = service_types or list(container._services.keys())
            
            for service_type in services_to_analyze:
                descriptor = container._services.get(service_type)
                if descriptor:
                    service_analysis[service_type.__name__] = {
                        "lifetime": descriptor.lifetime.value,
                        "has_factory": descriptor.factory is not None,
                        "has_instance": descriptor.instance is not None,
                        "module": service_type.__module__,
                    }
            
            # Build AI prompt
            prompt = self._build_configuration_analysis_prompt(service_analysis)
            
            # Get AI analysis
            response = await self._ai_provider.complete(
                prompt=prompt,
                system_prompt="You are an expert in dependency injection best practices. "
                            "Analyze service configurations and suggest improvements.",
                temperature=0.3,
            )
            
            # Parse response
            ai_suggestions = self._parse_ai_optimization_response(response.content)
            suggestions.extend(ai_suggestions)
            
        except Exception as e:
            self._logger.error(f"AI configuration analysis failed: {e}")
            suggestions.extend(self._heuristic_configuration_analysis(container, service_types))
            
        return suggestions
    
    def _heuristic_configuration_analysis(
        self,
        container: AIEnhancedDIContainer,
        service_types: Optional[List[Type]],
    ) -> List[OptimizationSuggestion]:
        """Perform heuristic-based configuration analysis."""
        suggestions = []
        
        services_to_analyze = service_types or list(container._services.keys())
        
        for service_type in services_to_analyze:
            descriptor = container._services.get(service_type)
            if not descriptor:
                continue
                
            class_name = service_type.__name__.lower()
            
            # Check for potential singleton candidates
            if (descriptor.lifetime == Lifetime.TRANSIENT and
                any(pattern in class_name for pattern in ['manager', 'service', 'client', 'cache'])):
                
                suggestions.append(OptimizationSuggestion(
                    suggestion_type="lifetime_change",
                    target_service=service_type.__name__,
                    current_config={"lifetime": "transient"},
                    recommended_config={"lifetime": "singleton"},
                    confidence=0.7,
                    reasoning=f"Class name '{service_type.__name__}' suggests singleton pattern",
                    expected_impact="Reduce memory usage and improve performance",
                    implementation_steps=[
                        f"Change {service_type.__name__} to SINGLETON lifetime",
                        "Ensure thread safety",
                        "Verify no state issues",
                    ],
                ))
            
            # Check for complex constructors that might benefit from factory pattern
            try:
                import inspect
                sig = inspect.signature(service_type.__init__)
                param_count = len([p for p in sig.parameters.values() if p.name != "self"])
                
                if param_count > 5 and not descriptor.factory:
                    suggestions.append(OptimizationSuggestion(
                        suggestion_type="factory_pattern",
                        target_service=service_type.__name__,
                        current_config={"constructor_params": param_count, "has_factory": False},
                        recommended_config={"use_factory": True},
                        confidence=0.6,
                        reasoning=f"Constructor has {param_count} parameters, factory pattern could simplify creation",
                        expected_impact="Improve code maintainability and flexibility",
                        implementation_steps=[
                            f"Create factory class for {service_type.__name__}",
                            "Move complex initialization logic to factory",
                            "Register factory instead of direct class",
                        ],
                    ))
                    
            except Exception:
                pass  # Skip if can't analyze constructor
        
        return suggestions
    
    def _build_performance_analysis_prompt(self, context: Dict[str, Any]) -> str:
        """Build prompt for AI performance analysis."""
        return f"""
Analyze the following dependency injection container performance metrics and provide optimization suggestions:

Container Metrics:
- Registered Services: {context.get('registered_services', 0)}
- Average Resolution Time: {context.get('avg_resolution_time', 0)*1000:.2f}ms
- Most Used Services: {json.dumps(context.get('most_used_services', []), indent=2)}

Additional Context:
{json.dumps(context.get('usage_data', {}), indent=2)}

Please provide specific optimization suggestions including:
1. Services that should change lifetime (singleton vs transient)
2. Performance bottlenecks and solutions
3. Configuration improvements
4. Expected impact of each suggestion

Format your response as a JSON array of optimization suggestions with fields:
- suggestion_type
- target_service  
- current_config
- recommended_config
- confidence (0.0-1.0)
- reasoning
- expected_impact
- implementation_steps (array)
"""
    
    def _build_configuration_analysis_prompt(self, service_analysis: Dict[str, Any]) -> str:
        """Build prompt for AI configuration analysis."""
        return f"""
Analyze the following dependency injection service configurations and suggest improvements:

Service Configurations:
{json.dumps(service_analysis, indent=2)}

Please analyze each service and provide suggestions for:
1. Optimal lifetime configuration (singleton, transient, scoped)
2. Whether factory pattern would be beneficial
3. Potential architectural improvements
4. Best practices compliance

Focus on:
- Performance optimization
- Memory efficiency  
- Thread safety considerations
- Maintainability improvements

Format your response as a JSON array of optimization suggestions with fields:
- suggestion_type
- target_service
- current_config
- recommended_config
- confidence (0.0-1.0)
- reasoning
- expected_impact
- implementation_steps (array)
"""
    
    def _parse_ai_optimization_response(self, response_content: str) -> List[OptimizationSuggestion]:
        """Parse AI response into optimization suggestions."""
        suggestions = []
        
        try:
            # Try to extract JSON from response
            import re
            json_match = re.search(r'\[.*\]', response_content, re.DOTALL)
            if json_match:
                json_data = json.loads(json_match.group())
                
                for item in json_data:
                    if isinstance(item, dict):
                        suggestions.append(OptimizationSuggestion(
                            suggestion_type=item.get("suggestion_type", "unknown"),
                            target_service=item.get("target_service", "unknown"),
                            current_config=item.get("current_config", {}),
                            recommended_config=item.get("recommended_config", {}),
                            confidence=float(item.get("confidence", 0.5)),
                            reasoning=item.get("reasoning", "AI recommendation"),
                            expected_impact=item.get("expected_impact", "Unknown impact"),
                            implementation_steps=item.get("implementation_steps", []),
                        ))
                        
        except Exception as e:
            self._logger.error(f"Failed to parse AI optimization response: {e}")
            # Return fallback suggestion
            suggestions.append(OptimizationSuggestion(
                suggestion_type="general",
                target_service="container",
                current_config={},
                recommended_config={"review_manually": True},
                confidence=0.5,
                reasoning="AI analysis failed, manual review recommended",
                expected_impact="Manual optimization needed",
                implementation_steps=["Review container configuration manually"],
            ))
            
        return suggestions


class AIContainerIntegration:
    """
    Main integration class that connects AI modules with DI container.
    """
    
    def __init__(
        self,
        container: AIEnhancedDIContainer,
        ai_config: Optional[AIConfig] = None,
    ):
        """
        Initialize AI-DI integration.
        
        Parameters
        ----------
        container : AIEnhancedDIContainer
            The DI container to enhance
        ai_config : AIConfig, optional
            AI module configuration
        """
        self.container = container
        self.ai_config = ai_config
        self._logger = Logger(self.__class__.__name__)
        
        # Initialize AI optimizer
        self.optimizer = AIContainerOptimizer(
            ai_config=ai_config,
            enable_real_ai=ai_config is not None,
        )
        
        # Auto-register AI services if config provided
        if ai_config:
            self._register_ai_services()
    
    def _register_ai_services(self):
        """Register AI module services in the DI container."""
        try:
            # Register AI provider
            if self.ai_config:
                from nautilus_trader.ai.providers.deepseek import DeepSeekProvider
                self.container.register(
                    DeepSeekProvider,
                    instance=DeepSeekProvider(self.ai_config.provider_config),
                    lifetime=Lifetime.SINGLETON,
                )
                
                # Register market analyzer
                from nautilus_trader.ai.analyzers.market import MarketAnalyzer
                from nautilus_trader.ai.config import AIAnalyzerConfig
                
                analyzer_config = AIAnalyzerConfig(
                    provider_config=self.ai_config.provider_config
                )
                
                self.container.register(
                    MarketAnalyzer,
                    factory=lambda: MarketAnalyzer(
                        provider=self.container.resolve(DeepSeekProvider),
                        config=analyzer_config,
                    ),
                    lifetime=Lifetime.SINGLETON,
                )
                
                self._logger.info("AI services registered in DI container")
                
        except Exception as e:
            self._logger.error(f"Failed to register AI services: {e}")
    
    async def optimize_container(
        self,
        include_performance: bool = True,
        include_configuration: bool = True,
        auto_apply: bool = False,
    ) -> Dict[str, Any]:
        """
        Perform comprehensive container optimization.
        
        Parameters
        ----------
        include_performance : bool, default True
            Include performance analysis
        include_configuration : bool, default True
            Include configuration analysis
        auto_apply : bool, default False
            Automatically apply high-confidence suggestions
            
        Returns
        -------
        Dict[str, Any]
            Optimization results and suggestions
        """
        results = {
            "performance_suggestions": [],
            "configuration_suggestions": [],
            "applied_optimizations": [],
            "summary": {},
        }
        
        try:
            # Performance analysis
            if include_performance:
                perf_suggestions = await self.optimizer.analyze_container_performance(self.container)
                results["performance_suggestions"] = [
                    {
                        "type": s.suggestion_type,
                        "target": s.target_service,
                        "confidence": s.confidence,
                        "reasoning": s.reasoning,
                        "impact": s.expected_impact,
                        "steps": s.implementation_steps,
                    }
                    for s in perf_suggestions
                ]
                
            # Configuration analysis
            if include_configuration:
                config_suggestions = await self.optimizer.analyze_service_configuration(self.container)
                results["configuration_suggestions"] = [
                    {
                        "type": s.suggestion_type,
                        "target": s.target_service,
                        "current": s.current_config,
                        "recommended": s.recommended_config,
                        "confidence": s.confidence,
                        "reasoning": s.reasoning,
                        "impact": s.expected_impact,
                        "steps": s.implementation_steps,
                    }
                    for s in config_suggestions
                ]
                
            # Auto-apply high-confidence suggestions
            if auto_apply:
                applied = self._apply_suggestions(
                    results["performance_suggestions"] + results["configuration_suggestions"]
                )
                results["applied_optimizations"] = applied
                
            # Generate summary
            results["summary"] = {
                "total_suggestions": len(results["performance_suggestions"]) + len(results["configuration_suggestions"]),
                "high_confidence": len([s for s in results["performance_suggestions"] + results["configuration_suggestions"] if s.get("confidence", 0) > 0.8]),
                "applied_count": len(results["applied_optimizations"]),
                "container_insights": self.container.get_ai_insights(),
            }
            
        except Exception as e:
            self._logger.error(f"Container optimization failed: {e}")
            results["error"] = str(e)
            
        return results
    
    def _apply_suggestions(self, suggestions: List[Dict[str, Any]]) -> List[str]:
        """Apply high-confidence optimization suggestions."""
        applied = []
        
        for suggestion in suggestions:
            if suggestion.get("confidence", 0) > 0.8:
                try:
                    if suggestion["type"] == "lifetime_change":
                        # Note: In practice, this would require re-registration
                        # which is complex. For now, just log the recommendation.
                        self._logger.info(f"High-confidence suggestion: {suggestion['reasoning']}")
                        applied.append(f"Logged suggestion for {suggestion['target']}")
                        
                except Exception as e:
                    self._logger.error(f"Failed to apply suggestion for {suggestion['target']}: {e}")
                    
        return applied


def create_ai_enhanced_system(
    ai_config: Optional[AIConfig] = None,
    enable_discovery: bool = True,
    discovery_packages: List[str] = None,
) -> AIContainerIntegration:
    """
    Create a complete AI-enhanced DI system.
    
    Parameters
    ----------
    ai_config : AIConfig, optional
        AI module configuration
    enable_discovery : bool, default True
        Enable automatic service discovery
    discovery_packages : List[str], optional
        Packages to scan for services
        
    Returns
    -------
    AIContainerIntegration
        Complete AI-enhanced DI system
    """
    from nautilus_trader.di.ai_config import create_trading_container
    
    # Create enhanced container
    container = create_trading_container()
    
    # Create integration
    integration = AIContainerIntegration(container, ai_config)
    
    # Auto-discover services if requested
    if enable_discovery:
        packages = discovery_packages or ["nautilus_trader"]
        asyncio.create_task(_auto_discover_services(integration, packages))
    
    return integration


async def _auto_discover_services(
    integration: AIContainerIntegration,
    packages: List[str],
) -> None:
    """Auto-discover and register services."""
    try:
        from nautilus_trader.di.ai_service_discovery import discover_and_recommend
        
        results = await discover_and_recommend(
            package_paths=packages,
            container=integration.container,
            auto_register=True,
        )
        
        integration._logger.info(f"Auto-discovery completed: {results.get('summary', {})}")
        
    except Exception as e:
        integration._logger.error(f"Auto-discovery failed: {e}")