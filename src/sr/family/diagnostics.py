"""Shared parent-action evidence for the production MC-only family workers."""
from __future__ import annotations

from collections.abc import Mapping

from inventory.policy.family import materialize_family_for_case
from sr.inventory.diagnostics import MonteCarloDiagnostics
from sr.core.diagnostics import action_signature

SHARED_DIAGNOSTICS = "__shared_diagnostics__"


class FamilyParentDiagnostics:
    """Resolve measured training cases and compare their fitted family actions."""

    def __init__(self, worker, domain):
        self.worker = worker
        self.domain = domain
        self.banks = {}
        self._fitted_actions = {}

    def _bank(self, case):
        if case.case_signature not in self.banks:
            self.banks[case.case_signature] = MonteCarloDiagnostics(
                case, int(self.worker.config.get("horizon", 1000)))
        return self.banks[case.case_signature]

    def _actions(self, parent, context_id, case_id, constants):
        if not isinstance(constants, Mapping):
            raise ValueError("diagnostic fitted constants are missing")
        if {item.name for item in parent.parameters} - constants.keys():
            raise ValueError("diagnostic fitted constants are incomplete")
        case = self._case(context_id, case_id)
        bank = self._bank(case)
        # Cache full vectors, never just the selected top-ten disagreement rows.
        key = (self.domain.semantic_hash(parent), bank.signature,
               tuple(sorted((name, float(value)) for name, value in constants.items())))
        if key not in self._fitted_actions:
            ast, motifs = materialize_family_for_case(parent, case)
            self._fitted_actions[key] = bank.actions(ast, constants, motifs)
        return bank, self._fitted_actions[key]

    def behavior_signature(self, evaluation):
        vectors = []
        for metric in sorted(evaluation.metrics, key=lambda row: (row.context_id, row.case_id)):
            bank, actions = self._actions(evaluation.family, metric.context_id, metric.case_id,
                                         metric.metadata.get("constants"))
            vectors.append([str(metric.context_id), metric.case_id, bank.signature, actions])
        if not vectors:
            raise ValueError("family behavior requires completed fitted case metrics")
        return action_signature(vectors)

    def _case(self, context_id, case_id):
        # Joint/P workers use the context verbatim; L uses numeric context ids.
        for name in (f"{context_id}__{case_id}", f"L{context_id}__{case_id}", case_id):
            if name in self.worker.case_map:
                return self.worker.case_map[name]
        raise ValueError(f"diagnostic training case is missing: {context_id}/{case_id}")

    def __call__(self, parents, evaluations):
        if not parents:
            return {}
        fitted = {}
        for parent in parents:
            key = self.domain.semantic_hash(parent)
            records = evaluations[key]["case_evaluations"]
            values = {}
            for record in records:
                pair = (str(record["context_id"]), str(record["case_id"]))
                constants = record.get("constants")
                if not isinstance(constants, Mapping):
                    raise ValueError(f"diagnostic fitted constants are missing: {key}/{pair}")
                if {item.name for item in parent.parameters} - constants.keys():
                    raise ValueError(f"diagnostic fitted constants are incomplete: {key}/{pair}")
                if pair in values:
                    raise ValueError(f"duplicate diagnostic fitting record: {key}/{pair}")
                values[pair] = constants
            fitted[key] = values
        pairs = set(next(iter(fitted.values())))
        if not pairs or any(set(records) != pairs for records in fitted.values()):
            raise ValueError("diagnostic parents must have the same nonempty set of fitted cases")
        diagnostics = []
        for context_id, case_id in sorted(pairs):
            case = self._case(context_id, case_id)
            bank = self._bank(case)
            actions = {}
            for parent in parents:
                key = self.domain.semantic_hash(parent)
                _, actions[key] = self._actions(parent, context_id, case_id,
                                                fitted[key][(context_id, case_id)])
            diagnostics.append({**bank.records(actions),
                                "context_id": context_id, "case_id": case_id})
        # Spread describes the whole parent set; send it once, rather than
        # duplicating every state/action record inside every parent's prompt row.
        shared = {"available": True, "regret_available": False,
                  "regret_unavailable_reason": "no_dp_reference", "cases": diagnostics}
        output = {self.domain.semantic_hash(parent): {
            "available": True, "regret_available": False,
            "regret_unavailable_reason": "no_dp_reference", "shared_records": SHARED_DIAGNOSTICS,
        } for parent in parents}
        output[SHARED_DIAGNOSTICS] = shared
        return output
