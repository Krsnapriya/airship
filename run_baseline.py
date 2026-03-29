import random
from typing import List, Optional
from models import Action, Observation  # pyre-ignore

def random_agent(obs: Observation) -> Action:
    """Returns a completely random valid action."""
    action_types = ["analyze_logs", "run_tests", "open_file", "edit_file"]
    atype = random.choice(action_types)
    
    target = None
    content = None
    
    if atype in ["open_file", "edit_file"]:
        # Mocking some targets since we don't have the full file list in Observation
        # (Observation only has visible_files, not all available files)
        # In a real scenario, the agent would know the filenames from previous steps.
        target = "utils.py" 
    
    if atype == "edit_file":
        content = "# Fixed bug\npass"
        
    return Action(type=atype, target=target, content=content)


class HeuristicAgent:
    """
    A deterministic heuristic agent for the Airship benchmark.
    Follows a logical debugging flow: Analyze -> Open -> Edit -> Test.
    """
    def __init__(self):
        self.last_action: Optional[str] = None
        self.current_target: Optional[str] = None

    def __call__(self, obs: Observation) -> Action:
        # 1. Start by analyzing logs if we have none or if we just ran tests
        if not obs.logs or self.last_action == "run_tests":
            self.last_action = "analyze_logs"
            return Action(type="analyze_logs")

        # 2. Extract potential bug location from logs (very simple heuristic)
        if "validator.py" in obs.logs:
            self.current_target = "validator.py"
        elif "service.py" in obs.logs:
            self.current_target = "service.py"
        else:
            self.current_target = "utils.py"

        # 3. If target not open, open it
        if self.current_target not in obs.visible_files:
            self.last_action = "open_file"
            return Action(type="open_file", target=self.current_target)

        # 4. If target open but not yet fixed (heuristic based on name)
        if self.last_action == "open_file":
            self.last_action = "edit_file"
            # Generic fix signature that environment recognizes
            content = "def fix(): pass # valid fixed" 
            return Action(type="edit_file", target=self.current_target, content=content)

        # 5. Finally, run tests
        self.last_action = "run_tests"
        return Action(type="run_tests")

if __name__ == "__main__":
    print("Airship Baseline Agents Loaded.")
