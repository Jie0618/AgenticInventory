"""Frozen training environments shared by all three stages."""
from experiments.training_inputs import load_training_cases


def cases(state_action_cap=20):
    return load_training_cases("lead_time_perishable", state_action_cap=state_action_cap)
