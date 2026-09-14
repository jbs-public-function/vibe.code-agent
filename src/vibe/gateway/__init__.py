"""OpenAI-compatible gateway (Phase 2, FR-03.x).

A fixed-address (`127.0.0.1:8080/v1`) OpenAI-compatible endpoint that reverse
proxies to whichever vLLM container the controller reports active, with an
optional swap-on-request mode. See
`.zed/docs/fable-5/features/03-openai-gateway.md` for the feature spec.
"""
