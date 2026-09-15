"""Zero-JavaScript admin UI (Phase 3, FR-01.x).

A browser dashboard on `http://localhost:8081`: model state and GPU stats,
load/unload with live job-progress pages, a chat client with server-side
sessions, and container log tailing. Everything is plain HTML forms,
meta-refresh, and chunked HTML — no JavaScript anywhere (NFR-01.1). All
controller access goes through `ControllerClient` (NFR-01.4). See
`.zed/docs/fable-5/features/01-admin-ui.md` for the feature spec.
"""
