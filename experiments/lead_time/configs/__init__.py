"""Frozen training environments shared by all three stages."""
from experiments.training_inputs import load_training_cases


def cases():
    return load_training_cases("lead_time")
