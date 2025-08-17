# External Dependency Warnings

This file documents warnings from external dependencies that are known to be harmless but may appear during execution or testing.

## weighs_norm deprecation
**Warning Message:**
```
/Users/Shared/Public/Github/bomdia/.venv/lib/python3.10/site-packages/torch/nn/utils/weight_norm.py:144: FutureWarning: `torch.nn.utils.weight_norm` is deprecated in favor of `torch.nn.utils.parametrizations.weight_norm`.
```

**Dependency:** Descript Audio Codec (version 1.0.0)
**Cause:** The warning is coming from the DAC (Descript Audio Codec) package, which is a dependency of the Dia model. The DAC package is using the deprecated torch.nn.utils.weight_norm function in several of its modules:
1. In **dac/nn/layers.py**, it imports `weight_norm` and uses it in helper functions: `WNConv1d`, `WNConvTranspose1d`.
2. In **dac/nn/quantize.py**, it imports `weight_norm` and uses the `WNConv1d`
     helper function (which uses `weight_norm`).
3. In **dac/model/discriminator.py**, it imports `weight_norm` and uses it in helper functions: `WNConv1d`, `WNConv2d`.

The warning is being triggered when the Dia model loads and initializes the DAC model in the `_load_dac_model` method. This happens when the Dia model is instantiated with `load_dac=True` (which is the default).

**Impact:** None - This is a harmless warning that does not affect functionality

**Status:** Expected behavior when using `decrypt-audio-codec` with torch 2.8.

**Last Updated:** August 17, 2025

## Pydantic v1/v2 Compatibility Warning

**Warning Message:**
```
PydanticSerializationUnexpectedValue(Expected 9 fields but got 6: Expected `Message` - serialized value may not be as expected [input_value=Message(content='1. **Ove...: None}, annotations=[]), input_type=Message])
PydanticSerializationUnexpectedValue(Expected `StreamingChoices` - serialized value may not be as expected [input_value=Choices(finish_reason='st...ider_specific_fields={}), input_type=Choices])
```

**Dependency:** LiteLLM (version 1.75.5.post1) and potentially langgraph

**Cause:** LiteLLM and langgraph are using deprecated Pydantic v1 patterns while the project uses Pydantic v2 (version 2.11.7)

**Impact:** None - This is a harmless warning that does not affect functionality

**Status:** Known issue, expected behavior when using LiteLLM with Pydantic v2

**Reference:** This is a common issue reported by other users of LiteLLM with Pydantic v2

**Resolution Plan:**
- Monitor future releases of LiteLLM and langgraph for Pydantic v2 full compatibility
- This warning can be safely ignored as it does not impact the application's functionality
- No action required in our codebase

**Last Updated:** August 11, 2025
