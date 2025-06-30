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
AI-powered service discovery and dependency analysis for DI container.
"""

import inspect
import importlib
import pkgutil
from typing import Any, Dict, List, Optional, Set, Type, Tuple
from dataclasses import dataclass
from pathlib import Path
import ast
import json

from nautilus_trader.common.component import Logger
from nautilus_trader.di.ai_enhanced_container import Injectable, Lifetime


@dataclass
class ServiceCandidate:
    """Represents a potential service for registration."""
    
    class_type: Type
    suggested_interface: Optional[Type]
    suggested_lifetime: Lifetime
    confidence: float
    reasoning: str
    dependencies: List[Type]
    is_abstract: bool
    module_path: str


@dataclass
class DependencyGraph:
    """Represents the dependency relationships between services."""
    
    nodes: Set[Type]
    edges: List[Tuple[Type, Type]]  # (dependent, dependency)
    cycles: List[List[Type]]
    complexity_score: float


class AIServiceDiscovery:
    """
    AI-powered service discovery that can automatically find and recommend
    services for registration based on code analysis.
    """
    
    def __init__(self, trusted_packages: List[str] = None):
        """
        Initialize AI service discovery.
        
        Parameters
        ----------
        trusted_packages : List[str], optional
            List of trusted package prefixes for discovery
        """
        self._logger = Logger(self.__class__.__name__)
        self._trusted_packages = trusted_packages or ["nautilus_trader"]
        self._discovered_services: Dict[str, ServiceCandidate] = {}
        
    def discover_services(
        self,
        package_paths: List[str],
        max_depth: int = 3,
        min_confidence: float = 0.6,
    ) -> List[ServiceCandidate]:
        """
        Discover potential services in the given packages.
        
        Parameters
        ----------
        package_paths : List[str]
            Package paths to scan for services
        max_depth : int, default 3
            Maximum recursion depth for package scanning
        min_confidence : float, default 0.6
            Minimum confidence threshold for recommendations
            
        Returns
        -------
        List[ServiceCandidate]
            List of discovered service candidates
        """
        candidates = []
        
        for package_path in package_paths:
            try:
                # Skip if not in trusted packages
                if not any(package_path.startswith(trusted) for trusted in self._trusted_packages):
                    self._logger.warning(f"Skipping untrusted package: {package_path}")
                    continue
                    
                package_candidates = self._scan_package(package_path, max_depth)
                candidates.extend(package_candidates)
                
            except Exception as e:
                self._logger.error(f"Error scanning package {package_path}: {e}")
                
        # Filter by confidence and deduplicate
        filtered = [c for c in candidates if c.confidence >= min_confidence]
        unique_candidates = self._deduplicate_candidates(filtered)
        
        self._logger.info(f"Discovered {len(unique_candidates)} service candidates")
        return unique_candidates
    
    def analyze_dependencies(self, candidates: List[ServiceCandidate]) -> DependencyGraph:
        """
        Analyze dependency relationships between service candidates.
        
        Parameters
        ----------
        candidates : List[ServiceCandidate]
            Service candidates to analyze
            
        Returns
        -------
        DependencyGraph
            Dependency graph with cycles and complexity analysis
        """
        nodes = {candidate.class_type for candidate in candidates}
        edges = []
        
        # Build dependency edges
        for candidate in candidates:
            for dependency in candidate.dependencies:
                if dependency in nodes:
                    edges.append((candidate.class_type, dependency))
                    
        # Detect cycles
        cycles = self._detect_cycles(nodes, edges)
        
        # Calculate complexity score
        complexity = self._calculate_complexity(nodes, edges, cycles)
        
        return DependencyGraph(
            nodes=nodes,
            edges=edges,
            cycles=cycles,
            complexity_score=complexity,
        )
    
    def generate_registration_recommendations(
        self,
        candidates: List[ServiceCandidate],
        dependency_graph: DependencyGraph,
    ) -> Dict[str, Any]:
        """
        Generate AI-powered recommendations for service registration.
        
        Parameters
        ----------
        candidates : List[ServiceCandidate]
            Service candidates
        dependency_graph : DependencyGraph
            Dependency analysis results
            
        Returns
        -------
        Dict[str, Any]
            Registration recommendations and insights
        """
        recommendations = {
            "high_priority": [],
            "medium_priority": [],
            "low_priority": [],
            "warnings": [],
            "optimizations": [],
        }
        
        # Categorize by priority
        for candidate in candidates:
            priority = self._determine_priority(candidate, dependency_graph)
            recommendations[priority].append({
                "class_name": candidate.class_type.__name__,
                "module": candidate.module_path,
                "interface": candidate.suggested_interface.__name__ if candidate.suggested_interface else None,
                "lifetime": candidate.suggested_lifetime.value,
                "confidence": candidate.confidence,
                "reasoning": candidate.reasoning,
            })
            
        # Add warnings for potential issues
        if dependency_graph.cycles:
            for cycle in dependency_graph.cycles:
                cycle_names = [cls.__name__ for cls in cycle]
                recommendations["warnings"].append(
                    f"Circular dependency detected: {' -> '.join(cycle_names)}"
                )
                
        if dependency_graph.complexity_score > 0.8:
            recommendations["warnings"].append(
                f"High dependency complexity (score: {dependency_graph.complexity_score:.2f}). "
                "Consider refactoring to reduce coupling."
            )
            
        # Generate optimization suggestions
        optimizations = self._generate_optimizations(candidates, dependency_graph)
        recommendations["optimizations"].extend(optimizations)
        
        return recommendations
    
    def _scan_package(self, package_path: str, max_depth: int) -> List[ServiceCandidate]:
        """Scan a package for service candidates."""
        candidates = []
        
        try:
            # Import the package
            package = importlib.import_module(package_path)
            
            # Scan for classes
            for item_name in dir(package):
                item = getattr(package, item_name)
                
                if inspect.isclass(item) and item.__module__.startswith(package_path):
                    candidate = self._analyze_class(item)
                    if candidate:
                        candidates.append(candidate)
                        
            # Recursively scan subpackages if depth allows
            if max_depth > 0 and hasattr(package, '__path__'):
                for _, subpackage_name, _ in pkgutil.iter_modules(package.__path__):
                    subpackage_path = f"{package_path}.{subpackage_name}"
                    try:
                        sub_candidates = self._scan_package(subpackage_path, max_depth - 1)
                        candidates.extend(sub_candidates)
                    except Exception as e:
                        self._logger.debug(f"Could not scan subpackage {subpackage_path}: {e}")
                        
        except Exception as e:
            self._logger.error(f"Error scanning package {package_path}: {e}")
            
        return candidates
    
    def _analyze_class(self, cls: Type) -> Optional[ServiceCandidate]:
        """Analyze a class to determine if it's a good service candidate."""
        try:
            # Skip if already marked as abstract
            if inspect.isabstract(cls):
                return None
                
            # Skip built-in types and non-instantiable classes
            if cls.__module__ in ('builtins', '__main__'):
                return None
                
            # Analyze class characteristics
            class_name = cls.__name__.lower()
            module_path = cls.__module__
            
            # Determine if it looks like a service
            confidence = 0.0
            reasoning = []
            suggested_lifetime = Lifetime.TRANSIENT
            
            # Check for service patterns
            service_patterns = [
                ('service', 0.8, Lifetime.SINGLETON),
                ('manager', 0.8, Lifetime.SINGLETON),
                ('client', 0.7, Lifetime.SINGLETON),
                ('handler', 0.7, Lifetime.TRANSIENT),
                ('processor', 0.7, Lifetime.TRANSIENT),
                ('analyzer', 0.6, Lifetime.TRANSIENT),
                ('factory', 0.6, Lifetime.TRANSIENT),
                ('builder', 0.5, Lifetime.TRANSIENT),
            ]
            
            for pattern, pattern_confidence, pattern_lifetime in service_patterns:
                if pattern in class_name:
                    confidence = max(confidence, pattern_confidence)
                    suggested_lifetime = pattern_lifetime
                    reasoning.append(f"Matches {pattern} pattern")
                    break
                    
            # Check if it inherits from Injectable
            if issubclass(cls, Injectable):
                confidence = max(confidence, 0.9)
                reasoning.append("Inherits from Injectable")
                
            # Analyze constructor
            dependencies = []
            try:
                sig = inspect.signature(cls.__init__)
                param_count = 0
                
                for name, param in sig.parameters.items():
                    if name == "self":
                        continue
                        
                    param_count += 1
                    
                    # Try to determine dependency type
                    if param.annotation != param.empty:
                        if inspect.isclass(param.annotation):
                            dependencies.append(param.annotation)
                            
                # Adjust confidence based on constructor complexity
                if param_count == 0:
                    confidence = max(confidence, 0.4)
                    reasoning.append("No dependencies (simple)")
                elif param_count <= 3:
                    confidence = max(confidence, 0.5)
                    reasoning.append("Few dependencies (manageable)")
                elif param_count > 5:
                    confidence *= 0.8  # Reduce confidence for complex constructors
                    reasoning.append("Many dependencies (complex)")
                    
            except Exception:
                reasoning.append("Could not analyze constructor")
                confidence *= 0.9
                
            # Find suggested interface
            suggested_interface = None
            for base in cls.__mro__[1:]:  # Skip the class itself
                if base != object and hasattr(base, '__abstractmethods__'):
                    suggested_interface = base
                    break
                    
            # Only return if confidence is reasonable
            if confidence >= 0.3:
                return ServiceCandidate(
                    class_type=cls,
                    suggested_interface=suggested_interface,
                    suggested_lifetime=suggested_lifetime,
                    confidence=confidence,
                    reasoning="; ".join(reasoning),
                    dependencies=dependencies,
                    is_abstract=inspect.isabstract(cls),
                    module_path=module_path,
                )
                
        except Exception as e:
            self._logger.debug(f"Error analyzing class {cls}: {e}")
            
        return None
    
    def _deduplicate_candidates(self, candidates: List[ServiceCandidate]) -> List[ServiceCandidate]:
        """Remove duplicate candidates, keeping the one with highest confidence."""
        seen = {}
        
        for candidate in candidates:
            key = candidate.class_type
            if key not in seen or candidate.confidence > seen[key].confidence:
                seen[key] = candidate
                
        return list(seen.values())
    
    def _detect_cycles(self, nodes: Set[Type], edges: List[Tuple[Type, Type]]) -> List[List[Type]]:
        """Detect cycles in the dependency graph."""
        # Simple cycle detection using DFS
        cycles = []
        visited = set()
        rec_stack = set()
        
        def dfs(node: Type, path: List[Type]):
            if node in rec_stack:
                # Found a cycle
                cycle_start = path.index(node)
                cycle = path[cycle_start:] + [node]
                cycles.append(cycle)
                return
                
            if node in visited:
                return
                
            visited.add(node)
            rec_stack.add(node)
            path.append(node)
            
            # Visit dependencies
            for source, target in edges:
                if source == node:
                    dfs(target, path.copy())
                    
            rec_stack.remove(node)
            
        for node in nodes:
            if node not in visited:
                dfs(node, [])
                
        return cycles
    
    def _calculate_complexity(
        self,
        nodes: Set[Type],
        edges: List[Tuple[Type, Type]],
        cycles: List[List[Type]],
    ) -> float:
        """Calculate dependency graph complexity score (0.0 to 1.0)."""
        if not nodes:
            return 0.0
            
        node_count = len(nodes)
        edge_count = len(edges)
        cycle_count = len(cycles)
        
        # Base complexity from connectivity
        if node_count <= 1:
            connectivity = 0.0
        else:
            max_edges = node_count * (node_count - 1)  # Max possible edges
            connectivity = edge_count / max_edges if max_edges > 0 else 0.0
            
        # Penalty for cycles
        cycle_penalty = min(cycle_count * 0.2, 0.5)
        
        # Penalty for high fan-out (nodes with many dependencies)
        fan_out_penalty = 0.0
        dependency_counts = {}
        for source, target in edges:
            dependency_counts[source] = dependency_counts.get(source, 0) + 1
            
        if dependency_counts:
            max_dependencies = max(dependency_counts.values())
            if max_dependencies > 5:
                fan_out_penalty = min((max_dependencies - 5) * 0.1, 0.3)
                
        complexity = min(connectivity + cycle_penalty + fan_out_penalty, 1.0)
        return complexity
    
    def _determine_priority(
        self,
        candidate: ServiceCandidate,
        dependency_graph: DependencyGraph,
    ) -> str:
        """Determine registration priority for a candidate."""
        # High priority: high confidence, no cycles, few dependencies
        if (candidate.confidence >= 0.8 and 
            candidate.class_type not in [cls for cycle in dependency_graph.cycles for cls in cycle] and
            len(candidate.dependencies) <= 3):
            return "high_priority"
            
        # Medium priority: decent confidence, manageable complexity
        elif candidate.confidence >= 0.6 and len(candidate.dependencies) <= 5:
            return "medium_priority"
            
        # Low priority: everything else
        else:
            return "low_priority"
    
    def _generate_optimizations(
        self,
        candidates: List[ServiceCandidate],
        dependency_graph: DependencyGraph,
    ) -> List[str]:
        """Generate optimization suggestions."""
        optimizations = []
        
        # Suggest singleton for stateless services with many dependents
        dependent_counts = {}
        for source, target in dependency_graph.edges:
            dependent_counts[target] = dependent_counts.get(target, 0) + 1
            
        for candidate in candidates:
            if (candidate.class_type in dependent_counts and 
                dependent_counts[candidate.class_type] > 3 and
                candidate.suggested_lifetime == Lifetime.TRANSIENT):
                optimizations.append(
                    f"Consider making {candidate.class_type.__name__} a SINGLETON "
                    f"(has {dependent_counts[candidate.class_type]} dependents)"
                )
                
        # Suggest factory pattern for complex constructors
        for candidate in candidates:
            if len(candidate.dependencies) > 5:
                optimizations.append(
                    f"Consider factory pattern for {candidate.class_type.__name__} "
                    f"(has {len(candidate.dependencies)} dependencies)"
                )
                
        return optimizations


def create_ai_discovery_session(trusted_packages: List[str] = None) -> AIServiceDiscovery:
    """Create an AI service discovery session."""
    return AIServiceDiscovery(trusted_packages or ["nautilus_trader"])


async def discover_and_recommend(
    package_paths: List[str],
    container: "AIEnhancedDIContainer" = None,
    auto_register: bool = False,
) -> Dict[str, Any]:
    """
    Discover services and generate recommendations.
    
    Parameters
    ----------
    package_paths : List[str]
        Packages to scan
    container : AIEnhancedDIContainer, optional
        Container to analyze current registrations
    auto_register : bool, default False
        Whether to automatically register high-confidence candidates
        
    Returns
    -------
    Dict[str, Any]
        Discovery results and recommendations
    """
    discovery = create_ai_discovery_session()
    
    # Discover candidates
    candidates = discovery.discover_services(package_paths)
    
    # Analyze dependencies
    dependency_graph = discovery.analyze_dependencies(candidates)
    
    # Generate recommendations
    recommendations = discovery.generate_registration_recommendations(
        candidates, dependency_graph
    )
    
    # Auto-register high-priority candidates if requested
    if auto_register and container:
        registered = []
        for rec in recommendations["high_priority"]:
            try:
                class_name = rec["class_name"]
                candidate = next(c for c in candidates if c.class_type.__name__ == class_name)
                
                container.register(
                    interface=candidate.suggested_interface or candidate.class_type,
                    implementation=candidate.class_type,
                    lifetime=candidate.suggested_lifetime,
                )
                registered.append(class_name)
                
            except Exception as e:
                discovery._logger.error(f"Failed to auto-register {class_name}: {e}")
                
        recommendations["auto_registered"] = registered
    
    # Add summary statistics
    recommendations["summary"] = {
        "total_candidates": len(candidates),
        "high_priority": len(recommendations["high_priority"]),
        "cycles_detected": len(dependency_graph.cycles),
        "complexity_score": dependency_graph.complexity_score,
    }
    
    return recommendations