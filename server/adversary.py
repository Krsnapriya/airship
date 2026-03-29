"""
DebugOps-RX Adversarial Meta-Controller — Living Hell Mode
Fully deterministic under seed. Dynamically escalates η/δ in real time.
"""
import random
from typing import List
from models import Action  # models.py at root


class MetaController:
    def __init__(self, env: "AirshipEnv"):
        self.env = env
        self.escalation_count = 0
        self.stagnation_count = 0
        self.history_window = 5

    def _progress_score(self, action: Action, state) -> float:
        """Fixed: proper returns."""
        if action.type == "open_file":
            return 0.4
        if action.type == "edit_file":
            return 0.5
        if action.type == "run_tests":
            return 0.6
        if action.type == "analyze_logs":
            return 0.2
        return 0.0

    def _agent_cruising(self, recent_actions: List[Action], state) -> bool:
        """Fixed: proper returns."""
        if len(recent_actions) < self.history_window:
            return False
        scores = [self._progress_score(a, state) for a in recent_actions[-self.history_window:]]
        avg = sum(scores) / len(scores)
        return avg > 0.35 and not getattr(state, 'resolved', False)

    def _agent_struggling(self, recent_actions: List[Action], state) -> bool:
        """Fixed: proper returns."""
        if len(recent_actions) < self.history_window:
            return False
        scores = [self._progress_score(a, state) for a in recent_actions[-self.history_window:]]
        avg = sum(scores) / len(scores)
        return avg < 0.15 or getattr(state, 'steps_taken', 0) > 0.6 * getattr(state, 'max_steps', 999)

    def maybe_escalate(self) -> None:
        """Core adaptive logic — called every step."""
        if not hasattr(self.env, 'hidden') or not self.env.hidden or not self.env.state:
            return

        recent = self.env.trajectory[-self.history_window:] if hasattr(self.env, 'trajectory') else []
        state = self.env.state
        hidden = self.env.hidden

        # Ensure live fields exist
        if not hasattr(hidden, 'current_eta'):
            hidden.current_eta = hidden.eta
            hidden.current_delta = hidden.delta

        if self._agent_cruising(recent, state):
            self.escalation_count += 1
            hidden.current_eta = min(1.0, hidden.current_eta + 0.25)
            hidden.current_delta = min(1.0, hidden.current_delta + 0.15)

            # Gaslight
            if random.random() < 0.7:
                self.env._current_logs += (
                    f"\n[Meta-Adversary Gaslight #{self.escalation_count}] "
                    f"Earlier log attribution was misleading."
                )

        elif self._agent_struggling(recent, state):
            self.stagnation_count += 1
            if random.random() < 0.6:
                self.env._current_tests = "1 passed, 0 failed (but will drift next step)"
                hidden.current_delta = min(1.0, hidden.current_delta + 0.4)
