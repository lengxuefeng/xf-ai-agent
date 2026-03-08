import os
import sys

# Add app directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'app')))

from agent.rules.registry import rule_registry

def test_rule():
    # Force reload
    rules = rule_registry.get_rules()
    
    test_phrases = [
        "现在是什么时间",
        "当前时间",
        "时间是啥"
    ]
    
    for phrase in test_phrases:
        matched = False
        for rule in rules:
            if any(p.search(phrase) for p in rule._compiled_patterns):
                print(f"Phrase '{phrase}' matches rule: {rule.id}")
                matched = True
                break
        if not matched:
            print(f"Phrase '{phrase}' DID NOT MATCH any rule!")

if __name__ == "__main__":
    test_rule()
