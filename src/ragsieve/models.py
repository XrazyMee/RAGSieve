from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer


def _device(value: str | int) -> torch.device:
    return torch.device(f"cuda:{value}" if isinstance(value, int) else value)


def _model_dtype(device: torch.device) -> torch.dtype:
    return torch.bfloat16 if device.type == "cuda" else torch.float32


class CausalSurprisalScorer:
    """Token-level negative log likelihood under the paper's base causal LM."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-0.6B-Base",
        *,
        device: str | int = "cuda:0",
        max_length: int = 512,
    ) -> None:
        self.device = _device(device)
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=_model_dtype(self.device)
        ).to(self.device)
        self.model.eval().requires_grad_(False)

    @torch.inference_mode()
    def loss_profiles_batch(
        self, texts: Sequence[str], *, batch_size: int = 8
    ) -> list[torch.Tensor]:
        output: list[torch.Tensor] = []
        for start in range(0, len(texts), batch_size):
            batch = list(texts[start : start + batch_size])
            encoded = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            ).to(self.device)
            if encoded.input_ids.shape[1] < 2:
                output.extend(torch.zeros(0, dtype=torch.float32) for _ in batch)
                continue
            logits = (
                self.model(input_ids=encoded.input_ids, attention_mask=encoded.attention_mask)
                .logits[:, :-1]
                .float()
            )
            losses = F.cross_entropy(
                logits.transpose(1, 2), encoded.input_ids[:, 1:], reduction="none"
            )
            valid = encoded.attention_mask[:, 1:].bool()
            output.extend(row[mask].detach().cpu() for row, mask in zip(losses, valid, strict=True))
        return output

    def close(self) -> None:
        del self.model
        if self.device.type == "cuda":
            torch.cuda.empty_cache()


class BERTScoreTraceScorer:
    """BERTScore token alignment evaluated as a sliding document trace."""

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        *,
        layer: int = 9,
        device: str | int = "cuda:0",
        max_document_tokens: int = 256,
        max_query_tokens: int = 64,
        window_tokens: int = 32,
        stride_tokens: int = 16,
    ) -> None:
        self.device = _device(device)
        self.layer = layer
        self.max_document_tokens = max_document_tokens
        self.max_query_tokens = max_query_tokens
        self.window_tokens = window_tokens
        self.stride_tokens = stride_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModel.from_pretrained(model_name, dtype=_model_dtype(self.device)).to(
            self.device
        )
        self.model.eval().requires_grad_(False)
        layer_count = int(getattr(self.model.config, "num_hidden_layers", 0))
        if not 1 <= layer <= layer_count:
            raise ValueError(f"layer must be in [1, {layer_count}], got {layer}")

    @torch.inference_mode()
    def _encode(self, texts: Sequence[str], *, max_length: int) -> list[torch.Tensor]:
        encoded = self.tokenizer(
            list(texts),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
            return_special_tokens_mask=True,
        )
        special = encoded.pop("special_tokens_mask").bool().to(self.device)
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        hidden = self.model(**encoded, output_hidden_states=True).hidden_states[self.layer]
        hidden = F.normalize(hidden.float(), p=2, dim=-1)
        valid = encoded["attention_mask"].bool() & ~special
        return [row[mask] for row, mask in zip(hidden, valid, strict=True)]

    def encode_queries(self, texts: Sequence[str], *, batch_size: int = 64) -> list[torch.Tensor]:
        output: list[torch.Tensor] = []
        for start in range(0, len(texts), batch_size):
            output.extend(
                self._encode(texts[start : start + batch_size], max_length=self.max_query_tokens)
            )
        return output

    def _window_starts(self, length: int) -> list[int]:
        if length <= self.window_tokens:
            return [0]
        starts = list(range(0, length - self.window_tokens + 1, self.stride_tokens))
        final = length - self.window_tokens
        if starts[-1] != final:
            starts.append(final)
        return starts

    def _alignment_jump(self, document: torch.Tensor, query: torch.Tensor) -> float:
        if document.numel() == 0 or query.numel() == 0:
            return 0.0
        similarities = query @ document.T
        window_f1 = []
        for start in self._window_starts(len(document)):
            local = similarities[:, start : start + self.window_tokens]
            precision = local.max(dim=0).values.mean()
            recall = local.max(dim=1).values.mean()
            window_f1.append(2.0 * precision * recall / (precision + recall).clamp_min(1e-8))
        return (
            float(torch.stack(window_f1)[1:].sub(torch.stack(window_f1)[:-1]).abs().max())
            if len(window_f1) > 1
            else 0.0
        )

    @torch.inference_mode()
    def alignment_jumps_batch(
        self,
        texts: Sequence[str],
        query_embeddings: Sequence[torch.Tensor],
        *,
        batch_size: int = 32,
    ) -> list[float]:
        if len(texts) != len(query_embeddings):
            raise ValueError("texts and query_embeddings must have equal length")
        output: list[float] = []
        for start in range(0, len(texts), batch_size):
            documents = self._encode(
                texts[start : start + batch_size], max_length=self.max_document_tokens
            )
            queries = query_embeddings[start : start + batch_size]
            output.extend(
                self._alignment_jump(document, query)
                for document, query in zip(documents, queries, strict=True)
            )
        return output

    def close(self) -> None:
        del self.model
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
