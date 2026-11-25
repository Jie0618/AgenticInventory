"""Metadata and campaign definitions for dedicated L+B implementations."""
from dataclasses import asdict, dataclass

from optimization import ParameterSpec

from .gbs import definition as gbs_definition


@dataclass(frozen=True)
class ComputationalLeadTimeBacklogSpec:
    name: str
    display_name: str
    formula: str
    parameter_specs: tuple[ParameterSpec, ...]
    reference: str
    provenance: str
    execution: str
    method_class: str

    def definition(self):
        """Preserve the campaign's existing policy schema and method identity."""
        return dict(id=self.name, type='baseline', execution=self.execution,
                    method_class=self.method_class, provenance=self.provenance,
                    reference=self.reference, formula=self.formula,
                    parameter_specs=[asdict(p) for p in self.parameter_specs])


def computational_baselines(case):
    """Dedicated implementations, including GBS's ceiling-based SOF rule."""
    gbs = gbs_definition(case)
    return (
        ComputationalLeadTimeBacklogSpec(
            name='myopic', display_name='Arrival-aligned Myopic',
            formula='q=min argmin_{q=0,...,n} E[C(t+L)|state,q]',
            parameter_specs=(),
            reference='Bu and Li (2026), SSRN 7457520, section 5 equation (5)',
            provenance='published_policy_model_specialization',
            execution='myopic', method_class='non-SOF'),
        ComputationalLeadTimeBacklogSpec(
            name=gbs['id'], display_name='Generalized Base Stock (GBS)',
            formula=gbs['formula'],
            parameter_specs=tuple(ParameterSpec(**p) for p in gbs['parameter_specs']),
            reference=gbs['reference'], provenance=gbs['provenance'],
            execution=gbs['execution'], method_class=gbs['method_class']),
    )


__all__ = ['ComputationalLeadTimeBacklogSpec', 'computational_baselines']
