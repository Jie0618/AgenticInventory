"""Load and validate frozen training environments for all four settings."""
import json
from pathlib import Path

from inventory.common.distributions import DiscreteDemand
from inventory.core.case import InventoryCase
from inventory.core.config import (BacklogConfig, InventoryModelConfig, LeadTimeConfig,
                                   PerishabilityConfig, TransitionSpec)


def training_payload(setting):
    path = Path(__file__).resolve().parent / setting / 'configs/training_environments.json'
    payload = json.loads(path.read_text())
    if payload.get('schema_version') != 1 or payload.get('setting') != setting:
        raise ValueError(f'invalid training environment schema: {path}')
    rows = payload['environments']
    names = [row['case_name'] for row in rows]
    if len(names) != 10 or len(set(names)) != 10:
        raise ValueError(f'training requires exactly ten distinct environments: {path}')
    return payload


def load_training_cases(setting, *, state_action_cap=None):
    output = []
    for row in training_payload(setting)['environments']:
        config = dict(row['config'])
        if config.pop('schema_version') != 1:
            raise ValueError('unsupported inventory configuration schema')
        config['lead_time'] = LeadTimeConfig(**config['lead_time'])
        config['perishability'] = PerishabilityConfig(**config['perishability'])
        config['backlog'] = BacklogConfig(**config['backlog'])
        config['transition'] = TransitionSpec(**config['transition'])
        case = InventoryCase(
            name=row['case_name'], config=InventoryModelConfig(**config),
            costs=row['costs'], demand=DiscreteDemand(**row['demand']),
            n=row['n'], kind=row['kind'], initial_state=row['initial_state'])
        if state_action_cap is not None and state_action_cap != case.n:
            config['capacity'] = state_action_cap
            case = InventoryCase(
                name=case.name, config=InventoryModelConfig(**config), costs=case.costs,
                demand=case.demand, n=state_action_cap, kind=case.kind,
                initial_state=case.initial_state.flatten(case.schema))
        output.append(case)
    return tuple(output)
