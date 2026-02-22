#!/usr/bin/env python3
"""
K 自動計算ロジックのテスト
"""

import sys
sys.path.insert(0, 'scripts')

from bevy_bridge import BevyBridge

# テスト 1: K_on_collocation を指定しない場合（自動計算）
print("=" * 60)
print("Test 1: Auto-calculate K_on_collocation")
print("=" * 60)

bridge1 = BevyBridge(
    strategy_type="strategy1",
    strategy_params={
        'collocation_resolution': 300,
    }
)
bridge1.initialize_domain(800, 800, epsilon=40.0)

print()

# テスト 2: K_on_collocation を明示的に指定する場合
print("=" * 60)
print("Test 2: Explicit K_on_collocation")
print("=" * 60)

bridge2 = BevyBridge(
    strategy_type="strategy1",
    strategy_params={
        'collocation_resolution': 300,
        'K_on_collocation': 3.5,
    }
)
bridge2.initialize_domain(800, 800, epsilon=40.0)
