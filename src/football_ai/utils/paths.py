import os
import sys

# Calculate absolute root path of the system (football-commentary-system/)
SYSTEM_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
THIRD_PARTY_SPORTS = os.path.join(SYSTEM_ROOT, "third_party", "sports")

def setup_third_party_paths():
    """Injects third_party/sports into sys.path if not already available."""
    if os.path.exists(THIRD_PARTY_SPORTS) and THIRD_PARTY_SPORTS not in sys.path:
        sys.path.insert(0, THIRD_PARTY_SPORTS)

# Run immediately on import for convenience
setup_third_party_paths()
