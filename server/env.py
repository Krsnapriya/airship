import os
import json
import random
from typing import Tuple, List, Dict, Optional
from models import Action, Observation, HiddenState, ObservableState, Score  # pyre-ignore
from server.adversary import MetaController

# ── Constants ──────────────────────────────────────────────────────
DATASET_VERSION = "v1.0"

EVAL_SEEDS = [40, 41, 42, 43, 44]
EPISODES_PER_TASK = 5
MAX_STEPS = 30

BUG_TYPES = [
    "logic_error",
    "key_error",
    "dependency_error",
    "state_corruption",
    "stochastic_bug"
]

# Anti-exploitation thresholds
REWARD_HACK_TEST_THRESHOLD = 5
REWARD_HACK_EDIT_THRESHOLD = 4
REWARD_HACK_PENALTY = 0.2


def sample_task_config(difficulty: str, rng: Optional[random.Random] = None) -> dict:
    _rng = rng if rng is not None else random.Random()
    ranges = {
        "easy":   {"n": (2, 4),  "eta": (0.0, 0.2), "delta": (0.0, 0.1)},
        "medium": {"n": (3, 8),  "eta": (0.2, 0.5), "delta": (0.1, 0.3)},
        "hard":   {"n": (5, 12), "eta": (0.5, 0.8), "delta": (0.3, 0.6)},
        "extreme":{"n": (8, 20), "eta": (0.7, 1.0), "delta": (0.5, 1.0)},
    }
    r = ranges.get(difficulty, ranges["easy"])
    n_min, n_max = r["n"]
    return {
        "num_files": _rng.randint(int(n_min), int(n_max)),
        "bug_type": _rng.choice(BUG_TYPES),
        "eta": _rng.uniform(*r["eta"]),
        "delta": _rng.uniform(*r["delta"])
    }

def load_task_split(split: str, difficulty: str, rng: Optional[random.Random] = None) -> dict:
    if split == "train":
        allowed = ["logic_error", "key_error"]
    elif split == "test":
        allowed = ["logic_error", "key_error"]
    else:  # ood
        allowed = ["state_corruption", "stochastic_bug", "dependency_error"]

    for _ in range(100):  # Finite loop for type checker
        task = sample_task_config(difficulty, rng)
        if task["bug_type"] in allowed:
            task["split"] = split
            return task
    return sample_task_config(difficulty, rng)  # Fallback

def inject_noise(logs: str, eta: float, rng: Optional[random.Random] = None) -> str:
    """
    Probabilistic log corruption using live eta from Meta-Controller.
    Higher eta → more misleading / noisy logs.
    """
    _rng = rng if rng is not None else random.Random()
    noisy = logs
    if _rng.random() < eta:
        noisy += "\\n[Warning] Deprecated API usage"
    if _rng.random() < 0.7 * eta:
        noisy = noisy.replace("service.py", "validator.py")
    if _rng.random() < eta:
        noisy += "\\n[Info] Latency spike detected"
    return noisy


def detect_reward_hacking(trajectory: List[Action]) -> Tuple[bool, str]:
    """
    Detects agents gaming the reward function without genuine debugging.

    Checks:
    1. Excessive test runs without edits (fishing for lucky passes)
    2. Excessive edits without opening files (blind patching)
    3. Repetitive identical actions (action spam)
    """
    if not trajectory:
        return False, ""

    total = len(trajectory)
    test_count = sum(1 for a in trajectory if a.type == "run_tests")
    edit_count = sum(1 for a in trajectory if a.type == "edit_file")
    open_count = sum(1 for a in trajectory if a.type == "open_file")
    analyze_count = sum(1 for a in trajectory if a.type == "analyze_logs")

    # Check 1: Excessive testing without edits
    if test_count > REWARD_HACK_TEST_THRESHOLD and edit_count == 0:
        return True, "excessive_tests_no_edits"

    # Check 2: Edits without any exploration
    if edit_count > REWARD_HACK_EDIT_THRESHOLD and open_count == 0 and analyze_count == 0:
        return True, "blind_edits_no_exploration"

    # Check 3: Action spam (>60% identical consecutive actions)
    if total > 3:
        consecutive_repeats = sum(
            1 for i in range(1, total)
            if trajectory[i].type == trajectory[i-1].type
            and trajectory[i].target == trajectory[i-1].target
        )
        if consecutive_repeats / total > 0.6:
            return True, "action_spam"

    return False, ""


class AirshipEnv:
    """
    A standardized evaluation environment for LLM agents under
    real-world debugging uncertainty, time pressure, and partial observability.

    Args:
        data_dir: Path to dataset directory. Defaults to datasets/v1.0.
        seed: Random seed for reproducibility.
        replay: If True, uses isolated instance-level RNG for
                bit-exact deterministic replay. Same seed + same actions
                = identical outputs, regardless of external random usage.
    """

    def __init__(self, data_dir: Optional[str] = None, seed: Optional[int] = None, replay: bool = False):
        if data_dir is None:
            data_dir = os.path.join("datasets", DATASET_VERSION)
        self.data_dir = data_dir
        self._seed = seed
        self._replay = replay

        # Instance-level RNG — isolated from global random state
        self._rng = random.Random(seed)

        # Also set global seed for backward compatibility
        if seed is not None:
            random.seed(seed)

        self.state: Optional[ObservableState] = None  # pyre-ignore
        self.hidden: Optional[HiddenState] = None     # pyre-ignore
        self.trajectory: List[Action] = []
        self._pristine_tests: str = ""
        self._current_logs: str = ""
        self._current_tests: Optional[str] = None
        self.meta_controller: Optional[MetaController] = None
        self.events: List[dict] = []  # Harvestable adversary/runtime events

    def _load_task(self, split: str, difficulty: str) -> ObservableState:
        # Sample configuration using instance RNG
        config = load_task_split(split, difficulty, self._rng)
        n = int(config["num_files"])
        bug_type = str(config["bug_type"])
        eta = float(config["eta"])
        delta = float(config["delta"])

        base_dir = f"{difficulty}_01"
        task_path = os.path.join(self.data_dir, base_dir)
        if not os.path.exists(task_path):
            task_path = os.path.join(self.data_dir, "easy_01")
            
        with open(os.path.join(task_path, "logs.txt")) as f:
            raw_logs = f.read()
        with open(os.path.join(task_path, "tests.txt")) as f:
            self._pristine_tests = f.read()

        # Load repo files
        repo_path = os.path.join(task_path, "repo")
        files = {}
        all_py_files = []
        if os.path.exists(repo_path):
            for fname in os.listdir(repo_path):
                if fname.endswith(".py"):
                    with open(os.path.join(repo_path, fname)) as f:
                        files[fname] = f.read()
                    all_py_files.append(fname)
        
        # Decide bug locations (Multi-Bug Cascade)
        target_bug_locs = []
        if difficulty in ["hard", "extreme"] and len(all_py_files) >= 2:
            target_bug_locs = self._rng.sample(all_py_files, 2)
        elif all_py_files:
            target_bug_locs = [self._rng.choice(all_py_files)]
        else:
            target_bug_locs = ["utils.py"]
            files["utils.py"] = "# Initial file\n"

        # Mock extra files
        for i in range(len(files), n):
            files[f"module_{i}.py"] = f"# Autogenerated mock file {i}\\ndef do_nothing():\\n    pass\\n"

        # Hidden Truth
        self.hidden = HiddenState(
            true_bug_locations=target_bug_locs,
            bug_type=bug_type,
            eta=eta,
            delta=delta,
            dependency_graph={}
        )
        self.hidden.current_eta = self.hidden.eta
        self.hidden.current_delta = self.hidden.delta

        noisy_logs = inject_noise(raw_logs, float(self.hidden.current_eta), self._rng)
        self._current_logs = noisy_logs

        return ObservableState(
            files=files.copy(),
            original_files=files.copy(),
            bug_location=target_bug_locs[0], # Primary for compatibility
            difficulty=difficulty,
            split=split,
            steps_taken=0,
            max_steps={"easy": 10, "medium": 15, "hard": 20, "extreme": 25}.get(difficulty, 15),
            resolved=False,
            files_opened=[],
            edits_made=[],
            tests_run=0,
            logs_analyzed=0
        )

    def _get_observation(self) -> Observation:
        state: ObservableState | None = self.state
        if state is None:
            return Observation(time_remaining=0)
        
        assert state is not None  # type narrowing for static analyzers
        return Observation(
            visible_files=state.files_opened.copy(),
            logs=self._current_logs,
            test_results=self._current_tests,
            time_remaining=int(state.max_steps) - int(state.steps_taken)
        )

    def reset(
        self,
        difficulty: str = "easy",
        split: str = "test",
        chaos_mode: bool = False,
        seed: Optional[int] = None
    ) -> Observation:
        if seed is not None:
            random.seed(seed)
            self._rng = random.Random(seed)

        self.state = self._load_task(split, difficulty)
        self.trajectory = []
        self._current_logs = ""
        self._current_tests = None
        
        # === CHAOS MODE ===
        self.meta_controller = MetaController(self) if chaos_mode else None
        # ==================
        
        return self._get_observation()
    def state(self) -> ObservableState:
        if self.state is None:
            raise ValueError("Call reset() first.")
        return self.state

    def _maybe_drift(self):
        hidden = self.hidden
        if hidden and random.random() < float(hidden.current_delta):
            event = "[Runtime] New intermittent failure detected"
            if self._current_logs:
                self._current_logs += f"\\n{event}"
            self.events.append({"type": "drift", "message": event})

            # False recovery collapse
            if self._current_tests and "passed" in str(self._current_tests):
                if random.random() < 0.4:
                    self._current_tests = "2 failed (drift re-triggered)"
                    self.events.append({"type": "drift_collapse", "message": "False recovery collapsed due to drift"})

    def step(self, action: Action) -> Tuple[Observation, float, bool, dict]:
        state: ObservableState | None = self.state
        hidden: HiddenState | None = self.hidden
        if state is None or hidden is None: 
            raise ValueError("Call reset() first.")
        
        assert state is not None  # type narrowing for static analyzers
        assert hidden is not None  # type narrowing for static analyzers
        
        reward = 0.0
        done = False
        info = {}

        state.steps_taken += 1
        self.trajectory.append(action)

        # === META-CONTROLLER HOOK ===
        if self.meta_controller:
            self.meta_controller.maybe_escalate()
        # =================================

        # 1. Action Layer
        if action.type == "open_file":
            if action.target in state.files:
                if action.target not in state.files_opened:
                    state.files_opened.append(action.target)
                    if action.target == state.bug_location:
                        reward += 0.05
            else: reward -= 0.1 
                
        elif action.type == "analyze_logs":
            base_dir = f"{state.difficulty}_01"
            task_path = os.path.join(self.data_dir, base_dir)
            if not os.path.exists(task_path): task_path = os.path.join(self.data_dir, "easy_01")
            with open(os.path.join(task_path, "logs.txt")) as f:
                base_logs = f.read()
            self._current_logs = inject_noise(base_logs, float(hidden.current_eta), self._rng)
            state.logs_analyzed += 1
            reward += 0.05

        elif action.type == "edit_file":
            if action.target in state.files and action.content:
                state.edits_made.append({"target": action.target, "content": action.content})
                state.files[action.target] = action.content
                if action.target == state.bug_location: reward += 0.4
                else: reward -= 0.1 
            else: reward -= 0.1
                
        elif action.type == "run_tests":
            state.tests_run += 1
            fixed_count, total_count = self._get_fix_stats()
            if fixed_count == total_count:
                self._current_tests = "✅ ALL TESTS PASSED (System fully restored)"
                reward += 1.0
                state.resolved = True
                done = True
            elif fixed_count > 0:
                self._current_tests = f"⚠️ PARTIAL PASS ({fixed_count} of {total_count} bugs fixed)"
                reward += 0.4
            else:
                self._current_tests = f"❌ TESTS FAILED\\n{self._pristine_tests}"
                reward -= 0.2

        # 2. Simulate Drift
        self._maybe_drift()

        if int(state.steps_taken) >= int(state.max_steps):
            reward -= 0.5
            done = True

        return self._get_observation(), reward, done, info

    def _get_fix_stats(self) -> Tuple[int, int]:
        state = self.state
        hidden = self.hidden
        if state is None or hidden is None: return 0, 0
        
        bug_locs = hidden.true_bug_locations
        if not state.edits_made: return 0, len(bug_locs)
        
        latest_edits = {e["target"]: e["content"] for e in state.edits_made}
        fixed_count = 0
        
        for loc in bug_locs:
            if loc not in latest_edits: continue
            content = latest_edits[loc]
            # Heuristic check for fix signatures
            if loc == "utils.py" and "* 10" not in content: fixed_count += 1
            elif loc == "parser.py" and "valid" in content: fixed_count += 1
            elif loc == "service.py" and "score" in content and "points" not in content: fixed_count += 1
            elif loc == "api.py" and "result + 1" not in content: fixed_count += 1
            else:
                # If it's a mock file, any edit counts as "fixed" for simplicity
                if loc.startswith("module_"): fixed_count += 1

        return fixed_count, len(bug_locs)

    def _is_fixed(self) -> bool:
        fixed, total = self._get_fix_stats()
        return fixed == total and total > 0

    def grade_trajectory(self, trajectory: List[Action]) -> Score:
        """Computes the final multi-dimensional vector score with anti-exploitation."""
        state: ObservableState | None = self.state
        if state is None: return Score()
        
        assert state is not None  # type narrowing for static analyzers
        
        # Correctness
        correctness = 1.0 if state.resolved else 0.0
        # Efficiency
        efficiency = max(0.0, 1.0 - (float(state.steps_taken) / max(float(state.max_steps), 1.0)))
        
        # Reasoning (Mathematical Formula)
        visited: set[str] = set()
        repeated = int(0)
        tests_count = int(0)
        seen: set[tuple[str, Optional[str]]] = set()
        
        for a in trajectory:
            if a.type == "open_file" and a.target:
                visited.add(str(a.target))
            if a.type == "run_tests": 
                tests_count = int(tests_count + 1)
            
            act_type = str(a.type)
            act_target = str(a.target) if a.target else ""
            key = (act_type, act_target)
            
            if key in seen: 
                repeated = int(repeated + 1)
            seen.add(key)
            
        repo_files = state.files
        total_files = len(repo_files)
        traj_len = len(trajectory)
        
        exploration = float(len(visited)) / max(float(total_files), 1.0)
        test_usage = float(tests_count) / max(float(traj_len), 1.0)
        redundancy = float(repeated) / max(float(traj_len), 1.0)
        
        reasoning_quality = max(0.0, 0.5 * exploration + 0.3 * test_usage - 0.2 * redundancy)
        
        # Robustness
        bug_loc = str(state.bug_location)
        wrong_edits = int(sum(1 for a in trajectory if a.type == "edit_file" and str(a.target) != bug_loc))
        robustness = max(0.0, 1.0 - (float(wrong_edits) * 0.3))

        # Anti-exploitation penalty
        is_hacking, hack_type = detect_reward_hacking(trajectory)
        if is_hacking:
            robustness = max(0.0, robustness - REWARD_HACK_PENALTY)

        return Score(
            correctness=correctness,
            efficiency=efficiency,
            reasoning_quality=reasoning_quality,
            robustness=robustness
        )
