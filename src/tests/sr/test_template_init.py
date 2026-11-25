from sr.family.template_init import PROTOCOL as TEMPLATE_PROTOCOL
import json
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from src.tests.support.family import family_domain as _domain_for
from src.tests.support.family import canonical_dual_shortfall_seed
from sr.family.template_init import initialize_templates


class TemplateInitTests(unittest.TestCase):
    def setUp(self):
        self.domain = _domain_for()
        self.seed = canonical_dual_shortfall_seed()
        self.other = replace(self.seed, family_id="second_seed", parameters=(), ast={"op": "const", "value": 99})

    def client(self, *, duplicates=False, failure_on=None):
        outer = self
        class Client:
            calls = []
            def complete(self, prompt):
                parents = json.JSONDecoder().raw_decode(prompt.split("\nRULE SEEDS: ")[1])[0]
                outer.assertEqual(len(parents), 1)
                parent, = parents
                key = parent['family_hash']
                self.calls.append(key)
                if key == failure_on:
                    raise RuntimeError("API failure")
                base = 10 if parent['family_id'] == outer.seed.family_id else 20
                return json.dumps({'families': [dict(
                    family_id=f"template_{index}", parent_family_hash=key,
                    ast={'op': 'const', 'value': (10 if duplicates else base) + index},
                    parameters=[], motifs={}, features=[]) for index in range(6)]})
        return Client()

    def test_each_seed_gets_its_own_six_templates_and_resume_reuses_calls(self):
        client = self.client()
        with TemporaryDirectory() as temporary:
            families, audit = initialize_templates(client, (self.seed, self.other),
                domain=self.domain, output_dir=Path(temporary))
            self.assertEqual(len(client.calls), 2)
            self.assertEqual(len(families), 12)
            self.assertEqual([r['accepted_count'] for r in audit['seeds']], [6, 6])
            self.assertEqual({f.parent_family_hash for f in families}, set(client.calls))
            again, _ = initialize_templates(client, (self.seed, self.other),
                domain=self.domain, output_dir=Path(temporary))
            self.assertEqual(again, families)
            self.assertEqual(len(client.calls), 2)
            with self.assertRaisesRegex(ValueError, 'five or six'):
                initialize_templates(client, (self.seed,), domain=self.domain,
                                     output_dir=Path(temporary), count=7)

    def test_cross_seed_duplicates_have_explicit_lineage_audit(self):
        with TemporaryDirectory() as temporary:
            families, audit = initialize_templates(self.client(duplicates=True),
                (self.seed, self.other), domain=self.domain, output_dir=Path(temporary))
            self.assertEqual(len(families), 6)
            self.assertEqual(audit['seeds'][1]['cross_seed_duplicates'], 6)
            self.assertEqual(audit['seeds'][1]['accepted_count'], 6)

    def test_failed_seed_is_audited_without_aborting_other_seed_templates(self):
        client = self.client(failure_on=self.domain.semantic_hash(self.other))
        with TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(RuntimeError, 'at least five'):
                initialize_templates(client, (self.seed, self.other), domain=self.domain,
                                     output_dir=Path(temporary))
            self.assertEqual(client.calls.count(self.domain.semantic_hash(self.seed)), 1)
            self.assertEqual(client.calls.count(self.domain.semantic_hash(self.other)), 3)
            self.assertEqual(len(list(Path(temporary).glob('template_init_seeds/*.json'))), 2)

    def test_resume_refills_exhausted_seed_without_recalling_completed_seeds(self):
        with TemporaryDirectory() as temporary:
            kwargs = dict(domain=self.domain, output_dir=Path(temporary))
            failed = self.client(failure_on=self.domain.semantic_hash(self.other))
            with self.assertRaisesRegex(RuntimeError, "at least five"):
                initialize_templates(failed, (self.seed, self.other), **kwargs)
            repaired = self.client()
            families, audit = initialize_templates(repaired, (self.seed, self.other), **kwargs)
            self.assertEqual(len(families), 12)
            self.assertEqual(repaired.calls, [self.domain.semantic_hash(self.other)])
            self.assertEqual(len(audit['seeds'][1]['attempts']), 4)

    def test_features_in_custom_motifs_are_materialized_before_validation(self):
        from inventory.policy.features import FeatureSpec
        family = replace(self.seed, parameters=(),
                         ast={'op': 'motif', 'name': 'custom_summary'},
                         motifs={'custom_summary': {'op': 'feature', 'name': 'custom_pipeline'}},
                         features=(FeatureSpec('custom_pipeline', 'head_sum', 1),))
        self.domain.validate_family(family)

    def test_partial_reply_is_refilled_with_existing_templates_in_prompt(self):
        base = self.client()
        class Partial:
            calls = 0
            def complete(self, prompt):
                self.calls += 1
                result = json.loads(base.complete(prompt))
                if self.calls == 1:
                    result['families'] = result['families'][:3]
                else:
                    self.has_existing = 'ALREADY ACCEPTED TEMPLATES' in prompt
                return json.dumps(result)
        client = Partial()
        with TemporaryDirectory() as temporary:
            families, audit = initialize_templates(client,(self.seed,),domain=self.domain,
                                                   output_dir=Path(temporary))
            self.assertEqual(len(families),6)
            self.assertEqual(client.calls,2)
            self.assertTrue(client.has_existing)

    def test_two_valid_templates_survive_exhausted_refill_budget(self):
        base = self.client()
        class Two:
            def complete(self, prompt):
                response = json.loads(base.complete(prompt))
                response['families'] = response['families'][:2]
                return json.dumps(response)
        with TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(RuntimeError, 'at least five'):
                initialize_templates(Two(), (self.seed,), domain=self.domain,
                                     output_dir=Path(temporary))
            saved, = Path(temporary).glob('template_init_seeds/*.json')
            payload = json.loads(saved.read_text())
            self.assertEqual(len(payload['families']), 2)
            self.assertEqual(len(payload['attempts']), 3)

    def test_strict_initializer_rejects_old_artifact_and_propagates_failures(self):
        from sr.family.search import MultiContextFamilySearch
        def initialize(_):
            raise RuntimeError('per seed failed')
        initialize.protocol = TEMPLATE_PROTOCOL
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            search = MultiContextFamilySearch(domain=self.domain,initial_families=(self.seed,),
                case_ids=tuple(str(i) for i in range(8)),worker=lambda *_:None,
                output_dir=root,initialization_fn=initialize)
            with self.assertRaisesRegex(RuntimeError,'per seed failed'):
                search._llm_initial_candidates()
            (root/'LLM_TEMPLATE_INIT.json').write_text(json.dumps({
                'protocol':'unsupported-initialization','families':[]}))
            with self.assertRaisesRegex(ValueError,'protocol mismatch'):
                search._llm_initial_candidates()

    def test_unsupported_initializers_fail_before_evaluation(self):
        from sr.family.search import MultiContextFamilySearch
        for protocol in (None, "unsupported-initialization"):
            with self.subTest(protocol=protocol), TemporaryDirectory() as temporary:
                def initialize(_):
                    self.fail("unsupported initializer was called")
                if protocol is not None:
                    initialize.protocol = protocol
                output = Path(temporary) / "run"
                with self.assertRaisesRegex(ValueError, "unsupported LLM initialization protocol"):
                    MultiContextFamilySearch(domain=self.domain, initial_families=(self.seed,),
                        case_ids=tuple(str(i) for i in range(8)),
                        worker=lambda *_: self.fail("worker was called"),
                        output_dir=output, initialization_fn=initialize)
                self.assertFalse(output.exists())

    def test_five_valid_templates_are_sufficient_without_extra_api_call(self):
        base = self.client()
        class Five:
            calls = 0
            def complete(self,prompt):
                self.calls += 1
                response = json.loads(base.complete(prompt))
                response['families'] = response['families'][:5]
                return json.dumps(response)
        client = Five()
        with TemporaryDirectory() as temporary:
            kwargs = dict(domain=self.domain,output_dir=Path(temporary))
            self.assertEqual(len(initialize_templates(client,(self.seed,),**kwargs)[0]),5)
            self.assertEqual(len(initialize_templates(client,(self.seed,),**kwargs)[0]),5)
            self.assertEqual(client.calls,1)


if __name__ == '__main__':
    unittest.main()
