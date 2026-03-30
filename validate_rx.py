import sys
import os

try:
    from server.env import AirshipEnv
    from server.adversary import MetaController
    from models import Action
    print("validate_rx: All Airship components loaded successfully.")
except ImportError as e:
    print(f"validate_rx: Missing dependency - {e}")
    sys.exit(1)

if __name__ == "__main__":
    print("Airship (DebugOps-RX) Validation Passed.")
