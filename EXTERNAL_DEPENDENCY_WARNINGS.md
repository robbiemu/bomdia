# External Dependency Warnings

This file documents warnings from external dependencies that are known to be harmless but may appear during execution or testing.

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
