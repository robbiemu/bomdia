to work with DIA without needing to revert our project, changes were made from the 61ae8e commit as follows:

git diff main -- pyproject.toml
diff --git a/pyproject.toml b/pyproject.toml
index dd844dd..d11675a 100644
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
-    "descript-audio-codec>=1.0.0",
-    "gradio>=5.25.2",
-    "huggingface-hub>=0.30.2",
+    "descript-audio-codec>=1.0.0", # Python >=3.10, sadly
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

 We also changed dia/model.py:
 - In two places there is `compute_dtype.to_dtype()` which makes a call on a dtype that is not valid with torch 2.8.
  line 440 was changed to:
  ```python
          # Apply CFG with efficient handling of special cases
        if cfg_scale == 0.0:
            # Pure unconditional generation - no guidance
            logits_BxCxV = uncond_logits_BxCxV
        elif cfg_scale == 1.0:
            # Pure conditional generation - no CFG computation needed
            logits_BxCxV = cond_logits_BxCxV
        else:
            # Standard CFG formula: start from unconditional baseline and apply guidance
            logits_BxCxV = uncond_logits_BxCxV + cfg_scale * (cond_logits_BxCxV - uncond_logits_BxCxV)
  ```
