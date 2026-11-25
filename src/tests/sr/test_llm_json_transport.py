"""Malformed SSE content must use the transport fallback, not empty proposals."""
import json
from unittest.mock import patch

import pytest

from sr.llm.client import JSONChatClient, LLMConfig


class Response:
    def __init__(self, content, *, stream=False, reason="stop"):
        self.content, self.stream, self.reason = content, stream, reason

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def __iter__(self):
        yield ('data: ' + json.dumps({'choices': [{'index': 0, 'delta': {'content': self.content}}]}) + '\n').encode()
        yield ('data: ' + json.dumps({'choices': [{'index': 0, 'delta': {}, 'finish_reason': self.reason}]}) + '\n').encode()
        yield b'data: [DONE]\n'

    def read(self):
        return json.dumps({'choices': [{'message': {'content': self.content}, 'finish_reason': self.reason}]}).encode()


def client(**kwargs):
    return JSONChatClient(LLMConfig('https://example.test/v1', 'secret', 'model',
                                  max_retries=1, **kwargs))


def test_done_stream_with_complete_mutations_recovers_without_fallback():
    bad = '{"features": [], "mutations": [{"action":"set_cap_mode"}],  '
    good = '{"features": [], "mutations": [{"action":"set_cap_mode"}]}'
    with patch('urllib.request.urlopen', side_effect=[Response(bad), Response(good)]) as request:
        assert json.loads(client().complete('test')) == json.loads(good)
    assert request.call_count == 1


@pytest.mark.parametrize('fields', [
    {"families": [{"ast": {"op": "const", "value": 0}}]},
    {"features": [], "mutations": [{"action": "set_cap_mode"}]},
    {"mutations": [{"action": "set_cap_mode"}], "features": [{"name": "tail"}]},
])
@pytest.mark.parametrize('suffix', [', ', ', "z_end_turn_review": "'])
def test_stage3_complete_business_fields_recovered(fields, suffix):
    raw = json.dumps(fields)[:-1] + suffix
    with patch('urllib.request.urlopen', return_value=Response(raw)) as request:
        assert json.loads(client().complete('test')) == fields
    assert request.call_count == 1


def test_invalid_stream_and_nonstream_are_not_reported_as_success(tmp_path):
    c = client()
    c.audit_log = tmp_path / 'audit.jsonl'
    with patch('urllib.request.urlopen', side_effect=[Response('{"mutations":['), Response('{"mutations":[')]):
        with pytest.raises(RuntimeError, match='invalid JSON'):
            c.complete('test')
    rows = [json.loads(s) for s in c.audit_log.read_text().splitlines()]
    assert all(r['status'] != 'completed' for r in rows)


def test_length_finish_is_rejected_even_when_json_parses():
    with patch('urllib.request.urlopen', side_effect=[Response('{}', reason='length'), Response('{}')]):
        assert client().complete('test') == '{}'


def test_invalid_nonstream_response_is_rejected():
    with patch('urllib.request.urlopen', return_value=Response('{"mutations":[],')):
        with pytest.raises(RuntimeError, match='invalid JSON'):
            client(stream=False).complete('test')


@pytest.mark.parametrize('suffix', [',  ', ', "z_end_turn_review": "'])
def test_complete_proposals_recovered_without_nonstream_request(tmp_path, suffix):
    c = client()
    c.audit_log = tmp_path / 'audit.jsonl'
    proposals = [{"action": "cap", "parent_hash": "abc"}]
    raw = '{"proposals": ' + json.dumps(proposals) + suffix
    with patch('urllib.request.urlopen', return_value=Response(raw)) as request:
        assert json.loads(c.complete('test')) == {"proposals": proposals}
    assert request.call_count == 1
    rows = [json.loads(line) for line in c.audit_log.read_text().splitlines()]
    assert rows[0]['status'] == 'stream_invalid'
    assert rows[0]['response'] == raw
    assert rows[1]['status'] == 'stream_metadata_recovered'


@pytest.mark.parametrize('raw', [
    '{"families": [{"ast":', '{"families": [],',
    '{"features": [],', '{"mutations": [{}],',
    '{"features": [], "mutations": [],',
    '{"features": [], "mutations": [42],',
    '{"features": [], "mutations": [{}], "families": [{}],',
    '{"features": [], "features": [], "mutations": [{}],',
    '{"families": [{}], "other_business_field": "',
    '{"proposals": [{"action": "cap"',
    '{"proposals": [],',
    '{"proposals": [42],',
    '{"proposals": [{}], "other_business_field": "',
    '{"proposals": [{}]',
])
def test_unknown_or_incomplete_business_content_not_recovered(raw):
    assert JSONChatClient._recover_proposals(raw, {'done': True, 'finish_reason': 'stop'}) is None


def test_recovery_requires_clean_transport_end():
    for metadata in [{}, {'done': True}, {'done': True, 'finish_reason': 'length'}]:
        assert JSONChatClient._recover_proposals('{"proposals": [{}],', metadata) is None


def test_fallback_disconnect_does_not_erase_stream_failure(tmp_path):
    from http.client import RemoteDisconnected
    c = client()
    c.audit_log = tmp_path / 'audit.jsonl'
    with patch('urllib.request.urlopen', side_effect=[Response('{"proposals":['), RemoteDisconnected('closed')]):
        with pytest.raises(RuntimeError):
            c.complete('test')
    rows = [json.loads(line) for line in c.audit_log.read_text().splitlines()]
    assert rows[0]['status'] == 'stream_invalid'
    assert rows[0]['response'] == '{"proposals":['
    assert rows[-1]['error_type'] == 'RemoteDisconnected'
