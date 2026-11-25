from __future__ import annotations

import json
import unittest

from inventory.policy.ast import ast_check, motif, parameter
from inventory.policy.motifs.lead_time import lead_time_motifs
from sr.llm.proposals import execute_structural_proposal, parse_structural_proposals


class SRLLMProposalTests(unittest.TestCase):
    def test_parses_and_executes_minimum_crossover(self) -> None:
        motifs = lead_time_motifs(3)
        checked = (
            ast_check(parameter("S"), motifs=motifs),
            ast_check(motif("pipeline_lag_moment"), motifs=motifs),
        )
        parents = {row.ast_hash: row.ast for row in checked}
        response = json.dumps(
            {
                "proposals": [
                    {
                        "action": "minimum_crossover",
                        "parent_hash": checked[0].ast_hash,
                        "other_parent_hash": checked[1].ast_hash,
                        "rationale": "combine a target with a lag constraint",
                    }
                ]
            }
        )

        proposal = parse_structural_proposals(response)[0]
        ast = execute_structural_proposal(proposal, parents, motifs)

        self.assertEqual(ast["op"], "minimum")
        self.assertIn("S", str(ast))
        self.assertIn("pipeline_lag_moment", str(ast))

    def test_accepts_double_encoded_proposal_object(self) -> None:
        inner = json.dumps(
            {"action": "replace_root", "ast": {"op": "const", "value": 1.0}}
        )
        proposals = parse_structural_proposals(
            json.dumps({"proposals": [inner]})
        )
        self.assertEqual(proposals[0].action, "replace_root")


if __name__ == "__main__":
    unittest.main()
