# Local LLM API Configuration

This directory contains only LLM role configuration templates. Fixed-policy evaluation, benchmarks, and plotting retained results do not require an API; the default LLM search requires the two local files below. See the [project guide](../README.md) for the complete workflow.

Copy the two role templates and fill in a Chat Completions-compatible service URL, API key, and model name:

```bash
cp ./configs/mutation_api_key.example.json ./configs/mutation_api_key.json
cp ./configs/motif_reflection_api_key.example.json ./configs/motif_reflection_api_key.json
```

Replace the `base_url`, `api_key`, and `model` placeholders in the local copies. The templates explicitly set sampling, timeout, and retry parameters; these values override the code defaults.

The complete three-stage entry point `run_stage123.sh` and each stage's `run.sh` read these two local files by default and all use LLMs. Set the `MUTATION_CONFIG` and `REFLECTION_CONFIG` environment variables to use different paths; when running an individual stage, you can also use `--mutation-config` and `--reflection-config`.

This directory does not control inventory models, training budgets, or MC evaluation budgets; those parameters are provided in the experiment directories. See [configuration and computing resources](../experiments/README.md#configuration-and-computing-resources) for their locations.

## Configuration Fields

| Field | Mutation template | Reflection template | Code default when omitted |
| --- | --- | --- | --- |
| `base_url` | Service URL placeholder | Service URL placeholder | Required; `/chat/completions` is appended automatically if absent |
| `api_key` | API key placeholder | API key placeholder | Required |
| `model` | Model name placeholder | Model name placeholder | Required |
| `temperature` | `0.7` | `0.4` | `0.4` |
| `timeout_seconds` | `300` | `300` | `150` |
| `max_retries` | `50` | `50` | `20` |
| `retry_window_seconds` | `3600` | `3600` | `1200` |
| `response_format_json` | `true` | `true` | `true` |
| `stream` | `true` | `true` | `true` |
| `retry_initial_delay_seconds` | Not set | Not set | `2` |
| `retry_max_delay_seconds` | Not set | Not set | `20` |
| `retry_backoff_multiplier` | Not set | Not set | `2` |

Timeouts and windows are measured in seconds. `timeout_seconds` controls the timeout for each request; `retry_window_seconds` controls the total retry window for one call and limits the remaining request time. `max_retries` is the maximum number of attempts, including the initial request; whichever limit is reached first applies.
