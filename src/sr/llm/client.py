"""Small OpenAI-compatible JSON client for the SR proposer."""

from __future__ import annotations

import json
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4


@dataclass(frozen=True)
class LLMConfig:
    base_url: str
    api_key: str
    model: str
    temperature: float = 0.4
    timeout_seconds: float = 150.0
    max_retries: int = 20
    retry_window_seconds: float = 1200.0
    retry_initial_delay_seconds: float = 2.0
    retry_max_delay_seconds: float = 20.0
    retry_backoff_multiplier: float = 2.0
    response_format_json: bool = True
    stream: bool = True

    def __post_init__(self) -> None:
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if self.max_retries < 1:
            raise ValueError("max_retries must be at least one")
        if self.retry_window_seconds <= 0:
            raise ValueError("retry_window_seconds must be positive")
        if self.retry_initial_delay_seconds < 0 or self.retry_max_delay_seconds < 0:
            raise ValueError("retry delays must be nonnegative")
        if self.retry_backoff_multiplier < 1:
            raise ValueError("retry_backoff_multiplier must be at least one")

    @classmethod
    def from_json(cls, path: Path) -> "LLMConfig":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            base_url=str(data["base_url"]),
            api_key=str(data["api_key"]),
            model=str(data["model"]),
            temperature=float(data.get("temperature", 0.4)),
            timeout_seconds=float(data.get("timeout_seconds", 150.0)),
            max_retries=int(data.get("max_retries", 20)),
            retry_window_seconds=float(data.get("retry_window_seconds", 1200.0)),
            retry_initial_delay_seconds=float(data.get("retry_initial_delay_seconds", 2.0)),
            retry_max_delay_seconds=float(data.get("retry_max_delay_seconds", 20.0)),
            retry_backoff_multiplier=float(data.get("retry_backoff_multiplier", 2.0)),
            response_format_json=bool(data.get("response_format_json", True)),
            stream=bool(data.get("stream", True)),
        )


class LLMHTTPError(RuntimeError):
    """HTTP failure carrying enough information to decide whether to retry."""

    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(f"LLM API HTTP {status_code}: {detail}")
        self.status_code = int(status_code)

    @property
    def retryable(self) -> bool:
        return self.status_code >= 500 or self.status_code in {408, 409, 425, 429}


class JSONChatClient:
    def __init__(
        self,
        config: LLMConfig,
        *,
        audit_log: Path | None = None,
        call_kind: str = "unspecified",
    ) -> None:
        self.config = config
        self.audit_log = None if audit_log is None else Path(audit_log)
        self.call_kind = str(call_kind)

    def complete(self, prompt: str) -> str:
        started = time.time()
        monotonic_started = time.monotonic()
        deadline = monotonic_started + self.config.retry_window_seconds
        call_id = uuid4().hex
        last_error: Exception | None = None
        attempts = 0
        for attempt_index in range(self.config.max_retries):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            attempts += 1
            try:
                response = self._complete_once(
                    prompt,
                    request_timeout_seconds=min(self.config.timeout_seconds, remaining),
                )
                self._validate_content(response)
                self._write_audit(
                    call_id=call_id,
                    prompt=prompt,
                    response=response,
                    started=started,
                    attempts=attempts,
                    status="completed",
                )
                self._write_progress(
                    call_id=call_id,
                    status="completed",
                    attempts=attempts,
                    started=started,
                )
                return response
            except (urllib.error.URLError, TimeoutError, OSError, KeyError, ValueError, RuntimeError) as exc:
                last_error = exc
                retryable = not isinstance(exc, LLMHTTPError) or exc.retryable
                can_retry = retryable and attempt_index + 1 < self.config.max_retries
                remaining = max(0.0, deadline - time.monotonic())
                delay = min(
                    self.config.retry_initial_delay_seconds
                    * self.config.retry_backoff_multiplier**attempt_index,
                    self.config.retry_max_delay_seconds,
                    remaining,
                )
                if not can_retry or remaining <= 0:
                    delay = 0.0
                self._write_audit(
                    call_id=call_id,
                    prompt=prompt,
                    response=None,
                    started=started,
                    attempts=attempts,
                    status="attempt_failed",
                    error_type=type(exc).__name__,
                    error_message=self._safe_error(exc),
                    retry_delay_seconds=delay,
                )
                self._write_progress(
                    call_id=call_id,
                    status="attempt_failed",
                    attempts=attempts,
                    started=started,
                    error_type=type(exc).__name__,
                    retry_delay_seconds=delay,
                )
                if not can_retry or remaining <= 0:
                    break
                if delay > 0:
                    time.sleep(delay)
        self._write_audit(
            call_id=call_id,
            prompt=prompt,
            response=None,
            started=started,
            attempts=attempts,
            status="failed",
            error_type=type(last_error).__name__ if last_error is not None else "unknown",
            error_message=self._safe_error(last_error),
        )
        self._write_progress(
            call_id=call_id,
            status="failed",
            attempts=attempts,
            started=started,
            error_type=type(last_error).__name__ if last_error is not None else "unknown",
        )
        elapsed = time.monotonic() - monotonic_started
        raise RuntimeError(
            f"LLM call failed after {attempts} attempt(s) and {elapsed:.1f}s: {last_error}"
        ) from last_error

    def _safe_error(self, error: Exception | None) -> str | None:
        if error is None:
            return None
        message = str(error)[:1000]
        if self.config.api_key:
            message = message.replace(self.config.api_key, "[REDACTED]")
        return message

    def _write_progress(
        self,
        *,
        call_id: str,
        status: str,
        attempts: int,
        started: float,
        error_type: str | None = None,
        retry_delay_seconds: float | None = None,
    ) -> None:
        """Emit a concise, secret-free line for tmux/tee monitoring."""

        row = {
            "progress": "llm",
            "time": time.time(),
            "call_id": call_id,
            "call_kind": self.call_kind,
            "status": status,
            "attempts": int(attempts),
            "elapsed_seconds": round(time.time() - started, 3),
        }
        if error_type is not None:
            row["error_type"] = error_type
        if retry_delay_seconds is not None:
            row["retry_delay_seconds"] = round(float(retry_delay_seconds), 3)
        print(json.dumps(row, sort_keys=True), file=sys.stdout, flush=True)

    def _write_audit(
        self,
        *,
        call_id: str,
        prompt: str,
        response: str | None,
        started: float,
        attempts: int,
        status: str,
        error_type: str | None = None,
        error_message: str | None = None,
        retry_delay_seconds: float | None = None,
    ) -> None:
        if self.audit_log is None:
            return
        self.audit_log.parent.mkdir(parents=True, exist_ok=True)
        row = {
            "schema_version": 1,
            "call_id": call_id,
            "call_kind": self.call_kind,
            "status": status,
            "started_at": started,
            "elapsed_seconds": time.time() - started,
            "attempts": int(attempts),
            "base_url": self.config.base_url,
            "model": self.config.model,
            "temperature": self.config.temperature,
            "stream": self.config.stream,
            "prompt": prompt,
            "response": response,
            "error_type": error_type,
            "error_message": error_message,
            "retry_delay_seconds": retry_delay_seconds,
        }
        # Deliberately omit api_key and request Authorization headers.
        with self.audit_log.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    def _complete_once(self, prompt: str, *, request_timeout_seconds: float) -> str:
        endpoint = self.config.base_url.rstrip("/")
        if not endpoint.endswith("/chat/completions"):
            endpoint += "/chat/completions"
        payload: dict[str, Any] = {
            "model": self.config.model,
            "temperature": self.config.temperature,
            "stream": self.config.stream,
            "messages": [
                {"role": "system", "content": "Return strictly valid JSON and no executable code."},
                {"role": "user", "content": prompt},
            ],
        }
        if self.config.response_format_json:
            payload["response_format"] = {"type": "json_object"}
        request = urllib.request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
                "Accept": "text/event-stream" if self.config.stream else "application/json",
                "User-Agent": "inventory-sof-sr/0.1",
            },
            method="POST",
        )
        request_started = time.monotonic()
        audit_started = time.time()
        transport_id = uuid4().hex
        try:
            with urllib.request.urlopen(request, timeout=request_timeout_seconds) as response:
                if self.config.stream:
                    content = None
                    metadata = {}
                    try:
                        content = self._read_stream(response, metadata=metadata)
                        self._validate_content(content)
                        return content
                    except (ValueError, OSError) as exc:
                        if content is None:
                            content = metadata.get("raw_content")
                        # Persist the original stream failure before a fallback
                        # can replace it with a different transport exception.
                        self._write_audit(call_id=transport_id, prompt=prompt, response=content,
                                          started=audit_started, attempts=1, status="stream_invalid",
                                          error_type=type(exc).__name__, error_message=self._safe_error(exc))
                        if isinstance(exc, OSError):
                            raise
                        recovered = self._recover_proposals(content, metadata)
                        if recovered is not None:
                            self._write_audit(call_id=transport_id, prompt=prompt, response=recovered,
                                              started=audit_started, attempts=1,
                                              status="stream_metadata_recovered")
                            return recovered
                        # Some OpenAI-compatible proxies advertise SSE but
                        # terminate it early or emit an empty stream. Retry
                        # the same request once as ordinary JSON; this avoids
                        # spending the whole retry window on a broken stream
                        # transport while preserving strict content checks.
                        remaining_timeout = request_timeout_seconds - (
                            time.monotonic() - request_started
                        )
                        if remaining_timeout <= 0:
                            raise
                        return self._complete_once_nonstream(
                            endpoint,
                            payload,
                            remaining_timeout,
                        )
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:1000]
            raise LLMHTTPError(exc.code, detail) from exc
        choice = data["choices"][0]
        if choice.get("finish_reason") in {"length", "content_filter"}:
            raise ValueError(f"LLM response terminated with {choice['finish_reason']}")
        content = choice["message"]["content"]
        self._validate_content(content)
        return content

    def _validate_content(self, content: str) -> None:
        """A terminal transport marker does not certify a complete JSON answer."""
        if not isinstance(content, str) or not content.strip():
            raise ValueError("LLM response completed without text content")
        if self.config.response_format_json:
            try:
                json.loads(content)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"LLM response is incomplete or invalid JSON at character {exc.pos}"
                ) from exc

    def _complete_once_nonstream(self, endpoint: str, payload: dict[str, Any], timeout: float) -> str:
        fallback = dict(payload)
        fallback["stream"] = False
        request = urllib.request.Request(
            endpoint,
            data=json.dumps(fallback).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": "inventory-sof-sr/0.1",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:1000]
            raise LLMHTTPError(exc.code, detail) from exc
        content = data.get("choices", [{}])[0].get("message", {}).get("content") if isinstance(data, dict) else None
        if not isinstance(content, str) or not content:
            raise ValueError("non-stream fallback completed without text content")
        reason = data["choices"][0].get("finish_reason")
        if reason in {"length", "content_filter"}:
            raise ValueError(f"LLM response terminated with {reason}")
        self._validate_content(content)
        return content

    @staticmethod
    def _recover_proposals(content: str | None, metadata: dict) -> str | None:
        """Extract only a complete business array before a known broken suffix.

        Never close an array, repair a candidate, or accept truncated transport.
        Ordinary proposal/schema/AST validation still runs downstream.
        """
        if content is None or metadata.get("finish_reason") != "stop" or not metadata.get("done"):
            return None
        # Each supported envelope has an explicit, complete business schema.
        # Stage3 uses families or features+mutations instead of Stage2 proposals.
        decoder = json.JSONDecoder()
        position = re.match(r'\s*\{\s*', content)
        if position is None:
            return None
        offset = position.end()
        fields = {}
        while True:
            try:
                name, end = decoder.raw_decode(content, offset)
            except ValueError:
                return None
            if (not isinstance(name, str)
                    or name not in {"proposals", "families", "features", "mutations"}
                    or name in fields):
                return None
            colon = re.match(r'\s*:\s*', content[end:])
            if colon is None:
                return None
            try:
                value, end = decoder.raw_decode(content, end + colon.end())
            except ValueError:
                return None
            if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
                return None
            fields[name] = value
            suffix = content[end:]
            if re.fullmatch(r'\s*,\s*(?:"z_end_turn_review"\s*:\s*")?\s*', suffix):
                valid = ((set(fields) == {"proposals"} and fields["proposals"])
                         or (set(fields) == {"families"} and fields["families"])
                         or (set(fields) == {"features", "mutations"} and fields["mutations"]))
                return json.dumps(fields) if valid else None
            separator = re.match(r'\s*,\s*', suffix)
            if separator is None:
                return None
            offset = end + separator.end()

    @staticmethod
    def _read_stream(response: Any, *, metadata: dict | None = None) -> str:
        """Read an OpenAI-compatible chat-completions SSE response.

        Requiring the terminal ``[DONE]`` marker is intentional. If a proxy
        drops an HTTP 200 stream halfway through a JSON answer, accepting the
        accumulated prefix would turn a transport failure into a malformed
        model response outside the retry loop.
        """

        parts: list[str] = []
        done = False
        for raw_line in response:
            line = raw_line.decode("utf-8").strip()
            if not line or line.startswith(":") or not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                done = True
                break
            chunk = json.loads(payload)
            choices = chunk.get("choices")
            if not isinstance(choices, list):
                raise ValueError("stream chunk has no choices list")
            for choice in choices:
                if not isinstance(choice, dict) or int(choice.get("index", 0)) != 0:
                    continue
                if choice.get("finish_reason") in {"length", "content_filter"}:
                    if metadata is not None:
                        metadata["raw_content"] = "".join(parts)
                    raise ValueError(f"LLM stream terminated with {choice['finish_reason']}")
                if metadata is not None and choice.get("finish_reason"):
                    metadata["finish_reason"] = choice["finish_reason"]
                delta = choice.get("delta")
                if not isinstance(delta, dict):
                    continue
                content = delta.get("content")
                if isinstance(content, str):
                    parts.append(content)
                elif isinstance(content, list):
                    for block in content:
                            if isinstance(block, dict) and isinstance(block.get("text"), str):
                                parts.append(block["text"])
                if metadata is not None:
                    metadata["raw_content"] = "".join(parts)
        if not done:
            raise ValueError("LLM stream ended before [DONE]")
        if metadata is not None:
            metadata["done"] = True
        output = "".join(parts)
        if not output:
            raise ValueError("LLM stream completed without text content")
        return output
