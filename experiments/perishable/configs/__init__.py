"""Frozen training environments shared by all three stages."""
from experiments.training_inputs import load_training_cases, training_payload


def cases():
    return list(load_training_cases("perishable"))


_metadata = training_payload("perishable")
HORIZON = _metadata["horizon"]
SUITE_NAME = _metadata["reference_suite"]
SELECTED_CASES = tuple(row["case_name"] for row in _metadata["environments"])
