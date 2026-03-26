"""Unified feature runtimes for SAE and transcoder features."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from jaxtyping import Float, Int
from torch import Tensor

from sae_lens import SAE, HookedSAETransformer
from transformer_lens import HookedTransformer

from circuit_tracer.utils.hf_utils import load_transcoder_from_hub


FeatureBackend = Literal["sae", "clt", "plt"]


@dataclass
class LayerActivations:
    """Token ids plus per-token feature activations for a single layer."""

    tokens: Int[Tensor, "batch seq"]
    attention_mask: Int[Tensor, "batch seq"]
    activations: Float[Tensor, "batch seq features"]


class BaseFeatureRuntime:
    """Common runtime interface for feature backends."""

    def __init__(self, model_name: str, device: str = "cuda", dtype: torch.dtype = torch.bfloat16):
        self.model_name = model_name
        self.device = device
        self.dtype = dtype
        self.model = None

    @property
    def tokenizer(self):
        self.load_model()
        return self.model.tokenizer

    def load_model(self):
        raise NotImplementedError

    def get_num_features(self, layer_index: int) -> int:
        raise NotImplementedError

    def get_layer_activations(
        self,
        texts: list[str],
        layer_index: int,
        max_length: int,
        feature_indices: list[int] | None = None,
        apply_activation_function: bool = True,
    ) -> LayerActivations:
        raise NotImplementedError

    def get_feature_max_activations(
        self,
        texts: list[str],
        layer_index: int,
        feature_index: int,
        max_length: int,
    ) -> np.ndarray:
        layer_acts = self.get_layer_activations(
            texts=texts,
            layer_index=layer_index,
            max_length=max_length,
            feature_indices=[feature_index],
            apply_activation_function=True,
        )
        max_acts = layer_acts.activations[..., 0].amax(dim=1)
        return max_acts.cpu().numpy()

    def get_single_feature_sequence(
        self,
        text: str,
        layer_index: int,
        feature_index: int,
        max_length: int,
    ) -> tuple[list[int], np.ndarray]:
        layer_acts = self.get_layer_activations(
            texts=[text],
            layer_index=layer_index,
            max_length=max_length,
            feature_indices=[feature_index],
            apply_activation_function=True,
        )
        seq_len = int(layer_acts.attention_mask[0].sum().item())
        token_ids = layer_acts.tokens[0, :seq_len].tolist()
        feature_acts = layer_acts.activations[0, :seq_len, 0].cpu().numpy()
        return token_ids, feature_acts


class SAEFeatureRuntime(BaseFeatureRuntime):
    """SAE-backed feature runtime."""

    def __init__(
        self,
        model_name: str,
        sae_name: str,
        sae_id_format: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__(model_name=model_name, device=device, dtype=dtype)
        self.sae_name = sae_name
        self.sae_id_format = sae_id_format
        self.sae = None
        self.current_layer = None

    def load_model(self):
        if self.model is None:
            self.model = HookedSAETransformer.from_pretrained_no_processing(
                self.model_name,
                device=self.device,
                dtype=self.dtype,
            )

    def load_sae(self, layer_index: int):
        if self.current_layer == layer_index and self.sae is not None:
            return

        sae_id = self.sae_id_format.format(layer=layer_index)
        self.sae = SAE.from_pretrained(
            release=self.sae_name,
            sae_id=sae_id,
            device=self.device,
        )
        if isinstance(self.sae, tuple):
            self.sae = self.sae[0]
        self.current_layer = layer_index

    def get_num_features(self, layer_index: int) -> int:
        self.load_model()
        self.load_sae(layer_index)
        return int(self.sae.cfg.d_sae)

    def get_layer_activations(
        self,
        texts: list[str],
        layer_index: int,
        max_length: int,
        feature_indices: list[int] | None = None,
        apply_activation_function: bool = True,
    ) -> LayerActivations:
        del apply_activation_function

        self.load_model()
        self.load_sae(layer_index)

        encoded = self.tokenizer(
            texts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)

        with torch.no_grad():
            _, cache = self.model.run_with_cache_with_saes(
                input_ids,
                saes=[self.sae],
                use_error_term=True,
            )

        try:
            hook_name = self.sae.cfg.metadata.hook_name
        except Exception:
            hook_name = self.sae.cfg.hook_name

        acts = cache[f"{hook_name}.hook_sae_acts_post"]
        if feature_indices is not None:
            acts = acts[:, :, feature_indices]

        acts = acts * attention_mask.unsqueeze(-1)
        return LayerActivations(
            tokens=input_ids.detach().cpu(),
            attention_mask=attention_mask.detach().cpu(),
            activations=acts.detach().cpu().float(),
        )


class TranscoderFeatureRuntime(BaseFeatureRuntime):
    """Runtime for CLT or PLT features loaded via circuit-tracer."""

    def __init__(
        self,
        model_name: str,
        transcoder_set: str,
        feature_backend: Literal["clt", "plt"],
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__(model_name=model_name, device=device, dtype=dtype)
        self.transcoder_set = transcoder_set
        self.feature_backend = feature_backend
        self.transcoders = None
        self.transcoder_config = None
        self.feature_input_hook = None

    def load_model(self):
        if self.model is None:
            self.model = HookedTransformer.from_pretrained(
                self.model_name,
                fold_ln=False,
                center_writing_weights=False,
                center_unembed=False,
                device=self.device,
                dtype=self.dtype,
            )

        if self.transcoders is None:
            device = torch.device(self.device)
            self.transcoders, self.transcoder_config = load_transcoder_from_hub(
                self.transcoder_set,
                device=device,
                dtype=self.dtype,
                lazy_encoder=False,
                lazy_decoder=True,
            )
            self.feature_input_hook = self.transcoders.feature_input_hook

    def get_num_features(self, layer_index: int) -> int:
        del layer_index
        self.load_model()
        return int(self.transcoders.d_transcoder)

    def get_layer_activations(
        self,
        texts: list[str],
        layer_index: int,
        max_length: int,
        feature_indices: list[int] | None = None,
        apply_activation_function: bool = True,
    ) -> LayerActivations:
        self.load_model()

        encoded = self.tokenizer(
            texts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        hook_name = f"blocks.{layer_index}.{self.feature_input_hook}"

        with torch.no_grad():
            _, cache = self.model.run_with_cache(input_ids, names_filter=[hook_name])
            hidden = cache[hook_name]
            acts = self.transcoders.encode_layer(
                hidden,
                layer_index,
                apply_activation_function=apply_activation_function,
            )

        if acts.ndim != 3:
            raise ValueError(f"Expected 3D activation tensor, got shape {tuple(acts.shape)}")

        if feature_indices is not None:
            acts = acts[:, :, feature_indices]

        # Ignore BOS position to match circuit-tracer's activation conventions.
        acts[:, :1, :] = 0
        acts = acts * attention_mask.unsqueeze(-1)

        return LayerActivations(
            tokens=input_ids.detach().cpu(),
            attention_mask=attention_mask.detach().cpu(),
            activations=acts.detach().cpu().float(),
        )


def build_feature_runtime(
    feature_backend: FeatureBackend,
    model_name: str,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    sae_name: str | None = None,
    sae_id_format: str | None = None,
    transcoder_set: str | None = None,
) -> BaseFeatureRuntime:
    """Create a runtime for the requested feature backend."""

    if feature_backend == "sae":
        if sae_name is None or sae_id_format is None:
            raise ValueError("SAE backend requires sae_name and sae_id_format")
        return SAEFeatureRuntime(
            model_name=model_name,
            sae_name=sae_name,
            sae_id_format=sae_id_format,
            device=device,
            dtype=dtype,
        )

    if transcoder_set is None:
        raise ValueError("Transcoder backends require transcoder_set")

    return TranscoderFeatureRuntime(
        model_name=model_name,
        transcoder_set=transcoder_set,
        feature_backend=feature_backend,
        device=device,
        dtype=dtype,
    )
