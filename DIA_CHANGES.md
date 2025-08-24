to work with DIA without needing to revert our project, changes were made from the 61ae8e commit as follows:

pyproject.toml:
```diff
diff --git a/pyproject.toml b/pyproject.toml
index dd844dd..fc7ac04 100644
--- a/pyproject.toml
+++ b/pyproject.toml
@@ -3,21 +3,27 @@ name = "nari-tts"
 version = "0.1.0"
 description = "Dia - A text-to-speech model for dialogue generation"
 readme = "README.md"
-requires-python = ">=3.10"
+requires-python = "==3.10"
 license = {file = "LICENSE"}
 authors = [
     {name = "Nari Labs", email = "contact@narilabs.ai"}
 ]
 dependencies = [
     "descript-audio-codec>=1.0.0",
-    "gradio>=5.25.2",
-    "huggingface-hub>=0.30.2",
+    "gradio~=5.42.0",
+    "huggingface-hub~=0.34.4",
+    "pydantic~=2.11.7",
+    "safetensors~=0.6.2",
+    "soundfile~=0.13.1",
+    "torch~=2.8.0",
+    "torchaudio~=2.8.0",
     "numpy>=2.2.4",
-    "pydantic>=2.11.3",
-    "safetensors>=0.5.3",
-    "soundfile>=0.13.1",
-    "torch==2.6.0",
-    "torchaudio==2.6.0",
+    "librosa>=0.10.0",      # 0.10+ already supports 3.10
+    "numba>=0.56",          # brings llvmlite>=0.39
+]
+
+[project.optional-dependencies]
+cuda = [
     "triton==3.2.0 ; sys_platform == 'linux'",
     "triton-windows==3.2.0.post18 ; sys_platform == 'win32'",
 ]
```

 We also changed dia/model.py:
```diff
diff --git a/dia/model.py b/dia/model.py
index 56a34ef..1a8b2d8 100644
--- a/dia/model.py
+++ b/dia/model.py
@@ -1,6 +1,6 @@
 import time
 from enum import Enum
-from typing import Callable
+from typing import Callable, List, Optional

 import numpy as np
 import torch
@@ -31,6 +31,7 @@ def _sample_next_token(
     top_p: float,
     top_k: int | None,
     audio_eos_value: int,
+    generators: Optional[List[torch.Generator]] = None,
 ) -> torch.Tensor:
     if temperature == 0.0:
         return torch.argmax(logits_BCxV, dim=-1)
@@ -71,8 +72,18 @@ def _sample_next_token(

     final_probs_BCxV = torch.softmax(logits_BCxV, dim=-1)

-    sampled_indices_BC = torch.multinomial(final_probs_BCxV, num_samples=1)
+    if generators is None:
+        sampled_indices_BC = torch.multinomial(final_probs_BCxV, num_samples=1)
+    else:
+        sampled_rows = []
+        for i, row in enumerate(final_probs_BCxV):
+            g = generators[i % len(generators)]  # cycle if more channels than batch
+            idx = torch.multinomial(row, num_samples=1, generator=g)
+            sampled_rows.append(idx)
+        sampled_indices_BC = torch.cat(sampled_rows, dim=0)
+
     sampled_indices_C = sampled_indices_BC.squeeze(-1)
+
     return sampled_indices_C


@@ -116,7 +127,7 @@ class Dia:
         self.device = device if device is not None else _get_default_device()
         if isinstance(compute_dtype, str):
             compute_dtype = ComputeDtype(compute_dtype)
-        self.compute_dtype = compute_dtype.to_dtype()
+        self.compute_dtype = compute_dtype
         self.model: DiaModel = DiaModel(config, self.compute_dtype)
         self.dac_model = None
         self._compiled_step = None
@@ -204,7 +215,7 @@ class Dia:

         # Load model directly using DiaModel's from_pretrained which handles HF download
         try:
-            loaded_model = DiaModel.from_pretrained(model_name, compute_dtype=compute_dtype.to_dtype())
+            loaded_model = DiaModel.from_pretrained(model_name, compute_dtype=compute_dtype)
         except Exception as e:
             raise RuntimeError(f"Error loading model from Hugging Face Hub ({model_name})") from e

@@ -405,6 +416,7 @@ class Dia:
         top_p: float,
         top_k: int,
         current_idx: int,
+        generators: Optional[List[torch.Generator]] = None,
     ) -> torch.Tensor:
         """Performs a single step of the decoder inference.

@@ -437,7 +449,7 @@ class Dia:

         uncond_logits_BxCxV = logits_last_Bx2xCxV[:, 0, :, :]  # Shape [B, C, V]
         cond_logits_BxCxV = logits_last_Bx2xCxV[:, 1, :, :]  # Shape [B, C, V]
-        logits_BxCxV = cond_logits_BxCxV + cfg_scale * (cond_logits_BxCxV - uncond_logits_BxCxV)
+        logits_BxCxV = uncond_logits_BxCxV + cfg_scale * (cond_logits_BxCxV - uncond_logits_BxCxV)

         _, top_k_indices_BxCxk = torch.topk(logits_BxCxV, k=top_k, dim=-1)
         mask_BxCxV = torch.ones_like(logits_BxCxV, dtype=torch.bool)
@@ -455,12 +467,17 @@ class Dia:

         flat_logits_BCxV = logits_BxCxV.view(B * self.config.decoder_config.num_channels, -1)

+        if generators is not None:
+            for rng in generators:
+                rng.initial_seed()
+
         pred_BC = _sample_next_token(
             flat_logits_BCxV.float(),
             temperature=temperature,
             top_p=top_p,
             top_k=top_k,
             audio_eos_value=audio_eos_value,
+            generators=generators,
         )

         pred_BxC = pred_BC.view(B, self.config.decoder_config.num_channels)
@@ -603,6 +620,7 @@ class Dia:
         audio_prompt: list[str | torch.Tensor | None] | str | torch.Tensor | None = None,
         audio_prompt_path: list[str | torch.Tensor | None] | str | torch.Tensor | None = None,
         use_cfg_filter: bool | None = None,
+        voice_seed: int | None = None,
         verbose: bool = False,
     ) -> np.ndarray | list[np.ndarray]:
         """Generates audio corresponding to the input text.
@@ -644,6 +662,17 @@ class Dia:
         delay_pattern_Cx = torch.tensor(delay_pattern, device=self.device, dtype=torch.long)
         self.model.eval()

+        # Create a deterministic generator for voice sampling once per call
+        gens = None
+        if voice_seed is not None:
+            B = len(text) if isinstance(text, list) else 1
+            gens = []
+            device = str(self.device or _get_default_device())
+            for _ in range(B):
+                g = torch.Generator(device=device)
+                g.manual_seed(int(voice_seed))
+                gens.append(g)
+
         if audio_prompt_path:
             print("Warning: audio_prompt_path is deprecated. Use audio_prompt instead.")
             audio_prompt = audio_prompt_path
@@ -703,13 +732,7 @@ class Dia:
             tokens_Bx1xC = dec_output.get_tokens_at(dec_step).repeat_interleave(2, dim=0)  # Repeat for CFG

             pred_BxC = self._decoder_step(
-                tokens_Bx1xC,
-                dec_state,
-                cfg_scale,
-                temperature,
-                top_p,
-                cfg_filter_top_k,
-                current_idx,
+                tokens_Bx1xC, dec_state, cfg_scale, temperature, top_p, cfg_filter_top_k, current_idx, generators=gens
             )

             current_idx += 1
```
