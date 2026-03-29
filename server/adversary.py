"""
Adversarial Meta-Controller — Reactive Escalation Policy
Dynamically adjusts η/δ based on agent behavior.
"""
import random
from typing import List, Optional
from models import Action, HiddenState  # models.py is at root


class MetaController:
    def __init__(self, env: "AirshipEnv"):
        self.env = env
        self.escalation_count = 0
        self.stagnation_count = 0
        self.history_window = 5

    def _progress_score(self, action: Action, state) -> float:
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
        if len(recent_actions) < self.history_window:
            return False
        scores = [self._progress_score(a, state) for a in recent_actions[-self.history_window:]]
        avg = sum(scores) / len(scores)
        return avg > 0.35 and not getattr(state, 'resolved', False)

    def _agent_struggling(self, recent_actions: List[Action], state) -> bool:
        if len(recent_actions) < self.history_window:
            return False
        scores = [self._progress_score(a, state) for a in recent_actions[-self.history_window:]]
        avg = sum(scores) / len(scores)
        return avg < 0.15 or getattr(state, 'steps_taken', 0) > 0.6 * getattr(state, 'max_steps', 999)

    def maybe_escalate(self) -> None:
        """Core adaptive logic — invoked every step."""
        if not hasattr(self.env, 'hidden') or not self.env.hidden or not self.env.state:
            return

        recent = self.env.trajectory[-self.history_window:] if hasattr(self.env, 'trajectory') else []
        state = self.env.state
        hidden = self.env.hidden

        if not hasattr(hidden, 'current_eta'):
            hidden.current_eta = hidden.eta
            hidden.current_delta = hidden.delta

        if self._agent_cruising(recent, state):
            self.escalation_count += 1
            hidden.current_eta = min(1.0, hidden.current_eta + 0.25)
            hidden.current_delta = min(1.0, hidden.current_delta + 0.15)
            self.env.events.append({"type": "trigger", "message": "High agent confidence → Escalation"})
            
            rng = getattr(self.env, '_rng', random)
            if rng.random() < 0.7:
                self.env.events.append({"type": "adversarial_noise", "message": "Log attribution was noisy. η↑ δ↑"})
                self.env._current_logs += f"\n[Adversary Context #{self.escalation_count}] Earlier log attribution was noisy."

        elif self._agent_struggling(recent, state):
            self.stagnation_count += 1
            rng = getattr(self.env, '_rng', random)
            if rng.random() < 0.6:
                self.env.events.append({"type": "trigger", "message": "Agent stagnation detected → Adversarial Trap"})
                # Signal recovery
                self.env._current_tests = "1 passed, 0 failed (environmental state change)"
                self.env.events.append({"type": "adversarial_noise", "message": "Temporary recovery signal (drift scheduled)"})
                hidden.current_delta = min(1.0, hidden.current_delta + 0.4)
