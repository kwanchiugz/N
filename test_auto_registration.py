#!/usr/bin/env python3
"""
Standalone test for auto-registration functionality.
This bypasses the Cython dependency issues to test the auto logic directly.
"""

import inspect
from typing import Any, Dict, List, Optional, Type, Union
from abc import ABC, abstractmethod
from enum import Enum
from dataclasses import dataclass


class Lifetime(Enum):
    SINGLETON = 'singleton'
    TRANSIENT = 'transient'
    SCOPED = 'scoped'


class Injectable(ABC):
    """Marker interface for injectable services."""
    pass


@dataclass
class AIRecommendation:
    """AI recommendation for service configuration."""
    recommended_lifetime: Lifetime
    confidence: float
    reasoning: List[str]
    potential_issues: List[str]


class MockLogger:
    """Mock logger to avoid nautilus dependencies."""
    def info(self, msg: str):
        print(f"INFO: {msg}")
    
    def warning(self, msg: str):
        print(f"WARNING: {msg}")
    
    def error(self, msg: str):
        print(f"ERROR: {msg}")


class AutoRegistrationTester:
    """Standalone tester for auto-registration logic."""
    
    def __init__(self):
        self._logger = MockLogger()
        self._enable_ai_recommendations = True
    
    def _get_ai_recommendation(self, service_type: Type) -> Optional[AIRecommendation]:
        """Get AI recommendation for service registration."""
        reasoning = []
        recommended_lifetime = Lifetime.TRANSIENT
        confidence = 0.6
        potential_issues = []
        
        # Analyze class characteristics
        class_name = service_type.__name__.lower()
        
        # Singleton patterns
        if any(pattern in class_name for pattern in ['manager', 'service', 'client', 'cache', 'pool']):
            recommended_lifetime = Lifetime.SINGLETON
            confidence = 0.8
            reasoning.append("Class name suggests singleton pattern (manager/service/client)")
            
        # Transient patterns  
        elif any(pattern in class_name for pattern in ['factory', 'builder', 'command', 'request']):
            recommended_lifetime = Lifetime.TRANSIENT
            confidence = 0.9
            reasoning.append("Class name suggests transient pattern (factory/builder/command)")
            
        # Scoped patterns
        elif any(pattern in class_name for pattern in ['context', 'session', 'transaction']):
            recommended_lifetime = Lifetime.SCOPED
            confidence = 0.8
            reasoning.append("Class name suggests scoped pattern (context/session/transaction)")
        
        # Analyze constructor complexity
        try:
            sig = inspect.signature(service_type.__init__)
            param_count = len([p for p in sig.parameters.values() if p.name != "self"])
            
            if param_count > 5:
                potential_issues.append(f"High constructor complexity ({param_count} dependencies)")
                confidence *= 0.8
                
            if param_count == 0 and confidence < 0.8:
                # Only suggest singleton for no dependencies if no strong pattern detected
                recommended_lifetime = Lifetime.SINGLETON
                reasoning.append("No dependencies suggests singleton safety")
                confidence = min(confidence + 0.1, 0.95)
                
        except Exception:
            potential_issues.append("Could not analyze constructor")
            confidence *= 0.9
        
        return AIRecommendation(
            recommended_lifetime=recommended_lifetime,
            confidence=confidence,
            reasoning=reasoning,
            potential_issues=potential_issues,
        )
    
    def test_auto_register(self, cls: Type) -> Dict[str, Any]:
        """Test auto-registration logic for a class."""
        result = {
            "class_name": cls.__name__,
            "success": False,
            "error": None,
            "recommended_lifetime": None,
            "confidence": 0.0,
            "reasoning": [],
            "issues": []
        }
        
        try:
            # Check for abstract classes (CRITICAL SAFETY CHECK)
            if inspect.isabstract(cls):
                raise TypeError(
                    f"Cannot auto-register '{cls.__name__}' because it is an abstract class. "
                    "Please register a concrete implementation for it explicitly."
                )
            
            # Get AI recommendation for lifetime
            recommended_lifetime = Lifetime.TRANSIENT  # Safe default
            
            if self._enable_ai_recommendations:
                ai_rec = self._get_ai_recommendation(cls)
                if ai_rec and ai_rec.confidence > 0.7:
                    recommended_lifetime = ai_rec.recommended_lifetime
                    result["confidence"] = ai_rec.confidence
                    result["reasoning"] = ai_rec.reasoning
                    result["issues"] = ai_rec.potential_issues
                    
                    self._logger.info(
                        f"AI auto-registration: {cls.__name__} as {recommended_lifetime.value} "
                        f"(confidence: {ai_rec.confidence:.2f})"
                    )
            
            result["recommended_lifetime"] = recommended_lifetime.value
            result["success"] = True
            
        except Exception as e:
            result["error"] = str(e)
            self._logger.error(f"Auto-registration failed for {cls.__name__}: {e}")
        
        return result


# Test classes
class DataService:
    """Should be recommended as SINGLETON."""
    def __init__(self):
        self.cache = {}


class OrderFactory:
    """Should be recommended as TRANSIENT."""
    def __init__(self):
        pass
    
    def create_order(self):
        return "Order"


class UserSessionContext:
    """Should be recommended as SCOPED."""
    def __init__(self, user_id: str):
        self.user_id = user_id


class ComplexAnalyzer:
    """Should have lower confidence due to complexity."""
    def __init__(self, data: str, config: dict, logger: Any, cache: dict, validator: Any, formatter: Any):
        self.data = data
        self.config = config
        self.logger = logger
        self.cache = cache
        self.validator = validator
        self.formatter = formatter


class AbstractHandler(ABC):
    """Should fail auto-registration (abstract)."""
    @abstractmethod
    def handle(self):
        pass


def main():
    """Run auto-registration tests."""
    print("🧪 Testing Auto-Registration Logic")
    print("=" * 50)
    
    tester = AutoRegistrationTester()
    
    test_classes = [
        DataService,
        OrderFactory, 
        UserSessionContext,
        ComplexAnalyzer,
        AbstractHandler,
    ]
    
    results = []
    for cls in test_classes:
        print(f"\n🔍 Testing: {cls.__name__}")
        result = tester.test_auto_register(cls)
        results.append(result)
        
        if result["success"]:
            print(f"✅ SUCCESS: {result['recommended_lifetime']} (confidence: {result['confidence']:.2f})")
            if result["reasoning"]:
                print(f"   Reasoning: {'; '.join(result['reasoning'])}")
            if result["issues"]:
                print(f"   ⚠️  Issues: {'; '.join(result['issues'])}")
        else:
            print(f"❌ FAILED: {result['error']}")
    
    # Summary
    print(f"\n📊 Test Summary")
    print("=" * 50)
    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]
    
    print(f"✅ Successful: {len(successful)}/{len(results)}")
    print(f"❌ Failed: {len(failed)}/{len(results)}")
    
    if successful:
        print("\nSuccessful registrations:")
        for r in successful:
            print(f"  • {r['class_name']}: {r['recommended_lifetime']} (confidence: {r['confidence']:.2f})")
    
    if failed:
        print("\nFailed registrations:")
        for r in failed:
            print(f"  • {r['class_name']}: {r['error']}")
    
    # Specific test cases
    print(f"\n🎯 Expected Results Verification:")
    expectations = [
        ("DataService", "singleton", "Service pattern should suggest singleton"),
        ("OrderFactory", "transient", "Factory pattern should suggest transient"),
        ("UserSessionContext", "scoped", "Context pattern should suggest scoped"),
        ("ComplexAnalyzer", "transient", "Complex constructor should have issues noted"),
        ("AbstractHandler", None, "Abstract class should fail registration"),
    ]
    
    for class_name, expected_lifetime, reason in expectations:
        result = next((r for r in results if r["class_name"] == class_name), None)
        if result:
            if result["success"]:
                actual = result["recommended_lifetime"]
                if actual == expected_lifetime:
                    print(f"✅ {class_name}: {actual} (as expected - {reason})")
                else:
                    print(f"⚠️  {class_name}: got {actual}, expected {expected_lifetime}")
            else:
                if expected_lifetime is None:
                    print(f"✅ {class_name}: correctly failed ({reason})")
                else:
                    print(f"❌ {class_name}: unexpected failure")
    
    print(f"\n🎉 Auto-registration logic test complete!")


if __name__ == "__main__":
    main()