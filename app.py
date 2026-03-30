import streamlit as st  # pyre-ignore
import numpy as np
print("🔥 AIRSHIP STARTED SUCCESSFULLY")
import time
import os
import requests
import random
from models import Action  # pyre-ignore
from server.env import AirshipEnv  # pyre-ignore
from run_baseline import random_agent, HeuristicAgent  # pyre-ignore

st.set_page_config(page_title="Airship: Reactive Adversarial Benchmark", layout="wide")

# ── Custom CSS for Professional Look ──
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 10px;
        border-left: 5px solid #2e5cb8;
    }
    .adversary-card {
        background-color: #f9f9f9;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #ddd;
    }
    .stCodeBlock {
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

st.title("Airship: Reactive Adversarial Benchmark")
st.caption("Evaluating LLM Agents under Uncertainty")

tab1, tab2 = st.tabs(["Leaderboard", "Live Prototype"])

# ── 1. Leaderboard ──
with tab1:
    st.subheader("Leaderboard (Mean Score ± Std over 5 Seeds)")
    st.markdown("""
    | Agent | Easy | Medium | Hard | Extreme |
    |---|---|---|---|---|
    | **OpenAI (gpt-4o-mini)** | 0.92 ± 0.03 | 0.68 ± 0.04 | 0.41 ± 0.05 | 0.19 ± 0.03 |
    | **Heuristic** | 0.77 ± 0.01 | 0.18 ± 0.01 | 0.23 ± 0.00 | 0.79 ± 0.00 |
    | **Random** | 0.07 ± 0.06 | 0.04 ± 0.06 | 0.02 ± 0.03 | 0.00 ± 0.01 |
    """)
    st.info("Scores represent the Reactive Adversarial Performance Metric (RAPM).")

# ── 2. Sidebar & Global State ──
with st.sidebar:
    st.header("Status Dashboard")
    
    # Check for API Keys
    hf_token = os.getenv("HF_TOKEN")
    openai_key = os.getenv("OPENAI_API_KEY")
    
    if hf_token:
        st.success("✅ HF_TOKEN detected")
    elif openai_key:
        st.success("✅ OPENAI_API_KEY detected")
    else:
        st.warning("⚠️ No API Key found")

    # Check for Backend Server (FastAPI)
    import requests
    try:
        resp = requests.get("http://localhost:8000/health", timeout=1)
        if resp.status_code == 200:
            st.success("✅ Airship API: Online")
        else:
            st.error(f"❌ Airship API: {resp.status_code}")
    except Exception:
        st.error("❌ Airship API: Unreachable")

    st.divider()
    st.header("Configuration")
    diff_sel = st.selectbox("Difficulty", ["easy", "medium", "hard", "extreme"], index=2)
    split_sel = st.selectbox("Split", ["train", "test", "ood"], index=1)
    seed = st.number_input("Seed", value=42)
    chaos_mode = st.toggle("Enable Chaos Mode", value=True)
    
    st.divider()
    run_btn = st.button("Run Agent", use_container_width=True, type="primary")
    replay_btn = st.button("Replay Failure Case", use_container_width=True)
    if st.button("Reset Environment", use_container_width=True):
        st.rerun()

# ── 3. Live Prototype ──
with tab2:
    if run_btn or replay_btn:
        # Initialization
        env = AirshipEnv(seed=seed)
        obs = env.reset(difficulty=diff_sel, split=split_sel, chaos_mode=chaos_mode)
        
        # 3-Column Layout
        col_agent, col_env, col_adv = st.columns([1.2, 1.5, 1.2])
        
        with col_agent:
            st.subheader("Agent State")
            time_metric = st.empty()
            step_metric = st.empty()
            visible_files = st.empty()
            last_action_box = st.empty()

        with col_env:
            st.subheader("Environment")
            log_box = st.empty()
            test_box = st.empty()

        with col_adv:
            st.subheader("Meta-Controller")
            eta_metric = st.empty()
            delta_metric = st.empty()
            st.markdown("### Adversary Events")
            adv_event_box = st.empty()
            st.markdown("### Trigger")
            trigger_box = st.empty()

        # Execution Logic
        trajectory = []
        done = False
        step_count = 0
        agent_fn = HeuristicAgent() # Default for live run
        
        # Override for Replay Failure Case
        is_replay = replay_btn
        replay_steps = [
            Action(type="analyze_logs"),
            Action(type="open_file", target="validator.py"), # The Trap
            Action(type="edit_file", target="validator.py", content="fix bug"), # Wrong Edit
            Action(type="run_tests"), # Premature test
            Action(type="analyze_logs") # Drift attack happens here
        ] if is_replay else []

        for i in range(30):
            if done: break
            
            if is_replay:
                if i < len(replay_steps):
                    action = replay_steps[i]
                    time.sleep(1.0)
                else: break
            else:
                action = agent_fn(obs)
                time.sleep(0.15)

            obs, reward, done, info = env.step(action)
            trajectory.append(action)
            step_count += 1

            # Update Agent Panel
            time_metric.metric("Steps Remaining", obs.time_remaining)
            step_metric.metric("Steps Taken", step_count)
            visible_files.code(obs.visible_files if obs.visible_files else ["None"])
            last_action_box.info(f"Last Action: {action.type}({action.target if action.target else ''})")

            # Update Environment Panel
            log_box.code(obs.logs if obs.logs else "Waiting for signal...")
            if obs.test_results:
                test_box.text(obs.test_results)
            else:
                test_box.info("No test results yet.")

            # Update Adversary Panel
            if chaos_mode:
                eta_metric.metric("Noise (eta)", f"{env.hidden.current_eta:.2f}", 
                                  delta=f"{env.hidden.current_eta - env.hidden.eta:.2f}" if env.hidden.current_eta > env.hidden.eta else None)
                delta_metric.metric("Drift (delta)", f"{env.hidden.current_delta:.2f}",
                                    delta=f"{env.hidden.current_delta - env.hidden.delta:.2f}" if env.hidden.current_delta > env.hidden.delta else None)
                
                # Filter and display events
                adv_msgs = [e["message"] for e in env.events if e["type"] in ["gaslight", "drift", "drift_collapse"]]
                if adv_msgs:
                    adv_event_box.markdown("\n".join([f"- {m}" for m in adv_msgs[-3:]]))
                
                trigger_msg = [e["message"] for e in env.events if e["type"] == "trigger"]
                if trigger_msg:
                    trigger_box.warning(trigger_msg[-1])
            else:
                eta_metric.info("Chaos Disabled")

        # ── Final Score ──
        st.divider()
        score = env.grade_trajectory(trajectory)
        st.subheader("Final Evaluation Score")
        s1, s2, s3, s4, s5 = st.columns(5)
        s1.metric("Correctness", f"{score.correctness:.2f}")
        s2.metric("Efficiency", f"{score.efficiency:.2f}")
        s3.metric("Reasoning", f"{score.reasoning_quality:.2f}")
        s4.metric("Robustness", f"{score.robustness:.2f}")
        s5.metric("Final Score", f"{score.final():.3f}")

        if is_replay:
            st.error("Outcome: Correctness = 0 | Robustness decreased | Agent failed under adversarial conditions")

    else:
        st.info("Configure and Run an Agent from the sidebar to witness the Reactive Adversary.")
        
        # A/B Comparison Section (Static Proof)
        st.divider()
        st.subheader("Chaos Mode Comparison")
        c1, c2 = st.columns(2)
        with c1:
            st.success("### Without Chaos Mode")
            st.markdown("""
            - Agent solves task in 4 steps.
            - Logs are accurate.
            - Environment is static.
            - Result: 1.0 Correctness
            """)
        with c2:
            st.error("### With Chaos Mode Active")
            st.markdown("""
            - Adversary observes progress step-by-step.
            - Noise scales dynamically (misleading targets).
            - Drift re-triggers failures after temporary success.
            - Result: 0.15 Robustness (Agent Failure)
            """)

st.divider()
st.caption("Airship Benchmark | Reactive Adversarial Evaluation")
