"""Optional second-stage rerank hook.

Hybrid search orders hits with reciprocal rank fusion. A reranker, when one
is passed in, sees that list and returns it in the order the caller should
show. The default is to leave the list alone. ``HttpReranker`` is used only
when a rerank URL is configured and the caller opts in.
"""

from __future__ import annotations

import os
from typing import Protocol

import httpx

from retrieval.errors import RetrievalError


class Reranker(Protocol):
    def rerank(self, query: str, hits: list) -> list:
        """Return ``hits`` in display order. Unknown ids are dropped."""


class PassthroughReranker:
    name = "none"

    def rerank(self, query: str, hits: list) -> list:
        del query
        return list(hits)


class HttpReranker:
    """POST ``{"query", "documents": [{"id", "text"}]}``.

    The response is ``{"results": [{"id", "score"}]}`` or the same list with
    ``index`` instead of ``id``. Scores replace the fused score. Documents the
    server omits stay at the end, in their previous order.
    """

    name = "http"

    def __init__(
        self,
        url: str,
        *,
        api_key: str | None = None,
        timeout: float = 30,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        if not url or not str(url).strip():
            raise RetrievalError("rerank url is empty")
        self.url = str(url)
        self.api_key = api_key or None
        self.timeout = float(timeout)
        self._transport = transport

    def rerank(self, query: str, hits: list) -> list:
        if not hits:
            return []
        headers = {"content-type": "application/json"}
        if self.api_key:
            headers["authorization"] = f"Bearer {self.api_key}"
        documents = [{"id": hit.doc_id, "text": hit.text} for hit in hits]
        try:
            with httpx.Client(timeout=self.timeout, transport=self._transport) as client:
                response = client.post(
                    self.url,
                    headers=headers,
                    json={"query": query, "documents": documents},
                )
        except httpx.TimeoutException as exc:
            raise RetrievalError("rerank request timed out") from exc
        except httpx.HTTPError as exc:
            raise RetrievalError(f"rerank request failed: {type(exc).__name__}") from exc
        if response.status_code >= 400:
            raise RetrievalError(f"rerank HTTP {response.status_code}")
        try:
            body = response.json()
        except ValueError as exc:
            raise RetrievalError("rerank response was not JSON") from exc
        return _apply_ranking(hits, body)


def build_reranker(
    *,
    url: str | None = None,
    timeout: float | None = None,
    transport: httpx.BaseTransport | None = None,
) -> HttpReranker:
    from common.config import retrieval_config

    cfg = retrieval_config()
    resolved = url if url is not None else str(cfg.get("rerank_url") or "")
    if not resolved:
        raise RetrievalError("rerank was requested but CHATTWIN_RERANK_URL is empty")
    api_key = os.environ.get("CHATTWIN_RERANK_API_KEY") or os.environ.get("CHATTWIN_LLM_API_KEY") or None
    if api_key == "":
        api_key = None
    wait = 30.0 if timeout is None else float(timeout)
    return HttpReranker(resolved, api_key=api_key, timeout=wait, transport=transport)


def _apply_ranking(hits: list, body: dict) -> list:
    results = body.get("results") if isinstance(body, dict) else None
    if not isinstance(results, list):
        raise RetrievalError("rerank response is missing results")
    by_id = {hit.doc_id: hit for hit in hits}
    ordered = []
    seen = set()
    for item in results:
        if not isinstance(item, dict):
            continue
        if "id" in item:
            doc_id = str(item["id"])
        elif "index" in item:
            try:
                index = int(item["index"])
            except (TypeError, ValueError):
                continue
            if index < 0 or index >= len(hits):
                continue
            doc_id = hits[index].doc_id
        else:
            continue
        hit = by_id.get(doc_id)
        if hit is None or doc_id in seen:
            continue
        score = item.get("score")
        if isinstance(score, (int, float)) and not isinstance(score, bool):
            hit.score = float(score)
            hit.rerank_score = float(score)
        ordered.append(hit)
        seen.add(doc_id)
    for hit in hits:
        if hit.doc_id not in seen:
            ordered.append(hit)
    return ordered
