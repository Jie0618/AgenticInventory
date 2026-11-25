"""Search-only structural bounds; executable baseline limits stay independent."""
from inventory.policy.ast import ASTLimits, ast_check
from inventory.policy.family import materialize_family_for_case
from sr.core.grammar import expand_motifs


def validate_search_family(family, domain):
    domain.validate_family(family)
    if len(family.parameters) > 8:
        raise ValueError("search families allow at most eight fitted parameters")
    for context in domain.contexts:
        ast, motifs = materialize_family_for_case(family, context.case)
        expanded = expand_motifs(ast, motifs)
        checked = ast_check(expanded, motifs={}, allowed_state_names=context.case.schema.names,
                            limits=ASTLimits(max_depth=10, max_nodes=64, max_parameters=8))
        if checked.stats.complexity > 150:
            raise ValueError("search expanded complexity exceeds 150")
