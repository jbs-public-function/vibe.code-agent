from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class ModelState(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    MODEL_STATE_UNSPECIFIED: _ClassVar[ModelState]
    AVAILABLE: _ClassVar[ModelState]
    LOADING: _ClassVar[ModelState]
    READY: _ClassVar[ModelState]
    UNLOADING: _ClassVar[ModelState]
    ERROR: _ClassVar[ModelState]

class JobKind(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    JOB_KIND_UNSPECIFIED: _ClassVar[JobKind]
    LOAD: _ClassVar[JobKind]
    UNLOAD: _ClassVar[JobKind]
    DOWNLOAD: _ClassVar[JobKind]
MODEL_STATE_UNSPECIFIED: ModelState
AVAILABLE: ModelState
LOADING: ModelState
READY: ModelState
UNLOADING: ModelState
ERROR: ModelState
JOB_KIND_UNSPECIFIED: JobKind
LOAD: JobKind
UNLOAD: JobKind
DOWNLOAD: JobKind

class ModelInfo(_message.Message):
    __slots__ = ("alias", "display_name", "hf_repo", "quantization", "max_model_len", "gpu_memory_utilization", "host_port", "tags", "use_case", "state", "cached_locally", "container_name", "extra_args", "yaml_preview")
    ALIAS_FIELD_NUMBER: _ClassVar[int]
    DISPLAY_NAME_FIELD_NUMBER: _ClassVar[int]
    HF_REPO_FIELD_NUMBER: _ClassVar[int]
    QUANTIZATION_FIELD_NUMBER: _ClassVar[int]
    MAX_MODEL_LEN_FIELD_NUMBER: _ClassVar[int]
    GPU_MEMORY_UTILIZATION_FIELD_NUMBER: _ClassVar[int]
    HOST_PORT_FIELD_NUMBER: _ClassVar[int]
    TAGS_FIELD_NUMBER: _ClassVar[int]
    USE_CASE_FIELD_NUMBER: _ClassVar[int]
    STATE_FIELD_NUMBER: _ClassVar[int]
    CACHED_LOCALLY_FIELD_NUMBER: _ClassVar[int]
    CONTAINER_NAME_FIELD_NUMBER: _ClassVar[int]
    EXTRA_ARGS_FIELD_NUMBER: _ClassVar[int]
    YAML_PREVIEW_FIELD_NUMBER: _ClassVar[int]
    alias: str
    display_name: str
    hf_repo: str
    quantization: str
    max_model_len: int
    gpu_memory_utilization: float
    host_port: int
    tags: _containers.RepeatedScalarFieldContainer[str]
    use_case: str
    state: ModelState
    cached_locally: bool
    container_name: str
    extra_args: _containers.RepeatedScalarFieldContainer[str]
    yaml_preview: str
    def __init__(self, alias: _Optional[str] = ..., display_name: _Optional[str] = ..., hf_repo: _Optional[str] = ..., quantization: _Optional[str] = ..., max_model_len: _Optional[int] = ..., gpu_memory_utilization: _Optional[float] = ..., host_port: _Optional[int] = ..., tags: _Optional[_Iterable[str]] = ..., use_case: _Optional[str] = ..., state: _Optional[_Union[ModelState, str]] = ..., cached_locally: _Optional[bool] = ..., container_name: _Optional[str] = ..., extra_args: _Optional[_Iterable[str]] = ..., yaml_preview: _Optional[str] = ...) -> None: ...

class ListModelsRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class ListModelsResponse(_message.Message):
    __slots__ = ("models",)
    MODELS_FIELD_NUMBER: _ClassVar[int]
    models: _containers.RepeatedCompositeFieldContainer[ModelInfo]
    def __init__(self, models: _Optional[_Iterable[_Union[ModelInfo, _Mapping]]] = ...) -> None: ...

class GetStatusRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class Status(_message.Message):
    __slots__ = ("active_alias", "state", "container_id", "host_port", "started_at_unix", "current_job", "last_error")
    ACTIVE_ALIAS_FIELD_NUMBER: _ClassVar[int]
    STATE_FIELD_NUMBER: _ClassVar[int]
    CONTAINER_ID_FIELD_NUMBER: _ClassVar[int]
    HOST_PORT_FIELD_NUMBER: _ClassVar[int]
    STARTED_AT_UNIX_FIELD_NUMBER: _ClassVar[int]
    CURRENT_JOB_FIELD_NUMBER: _ClassVar[int]
    LAST_ERROR_FIELD_NUMBER: _ClassVar[int]
    active_alias: str
    state: ModelState
    container_id: str
    host_port: int
    started_at_unix: int
    current_job: Job
    last_error: str
    def __init__(self, active_alias: _Optional[str] = ..., state: _Optional[_Union[ModelState, str]] = ..., container_id: _Optional[str] = ..., host_port: _Optional[int] = ..., started_at_unix: _Optional[int] = ..., current_job: _Optional[_Union[Job, _Mapping]] = ..., last_error: _Optional[str] = ...) -> None: ...

class Job(_message.Message):
    __slots__ = ("id", "kind", "alias", "started_at_unix", "done", "ok", "terminal", "last_error", "started_at_unix_ms", "finished_at_unix_ms")
    ID_FIELD_NUMBER: _ClassVar[int]
    KIND_FIELD_NUMBER: _ClassVar[int]
    ALIAS_FIELD_NUMBER: _ClassVar[int]
    STARTED_AT_UNIX_FIELD_NUMBER: _ClassVar[int]
    DONE_FIELD_NUMBER: _ClassVar[int]
    OK_FIELD_NUMBER: _ClassVar[int]
    TERMINAL_FIELD_NUMBER: _ClassVar[int]
    LAST_ERROR_FIELD_NUMBER: _ClassVar[int]
    STARTED_AT_UNIX_MS_FIELD_NUMBER: _ClassVar[int]
    FINISHED_AT_UNIX_MS_FIELD_NUMBER: _ClassVar[int]
    id: str
    kind: JobKind
    alias: str
    started_at_unix: int
    done: bool
    ok: bool
    terminal: bool
    last_error: str
    started_at_unix_ms: int
    finished_at_unix_ms: int
    def __init__(self, id: _Optional[str] = ..., kind: _Optional[_Union[JobKind, str]] = ..., alias: _Optional[str] = ..., started_at_unix: _Optional[int] = ..., done: _Optional[bool] = ..., ok: _Optional[bool] = ..., terminal: _Optional[bool] = ..., last_error: _Optional[str] = ..., started_at_unix_ms: _Optional[int] = ..., finished_at_unix_ms: _Optional[int] = ...) -> None: ...

class GetJobRequest(_message.Message):
    __slots__ = ("job_id",)
    JOB_ID_FIELD_NUMBER: _ClassVar[int]
    job_id: str
    def __init__(self, job_id: _Optional[str] = ...) -> None: ...

class JobEvent(_message.Message):
    __slots__ = ("job_id", "ts_unix_ms", "level", "message", "state", "progress", "terminal", "kind", "alias")
    JOB_ID_FIELD_NUMBER: _ClassVar[int]
    TS_UNIX_MS_FIELD_NUMBER: _ClassVar[int]
    LEVEL_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    STATE_FIELD_NUMBER: _ClassVar[int]
    PROGRESS_FIELD_NUMBER: _ClassVar[int]
    TERMINAL_FIELD_NUMBER: _ClassVar[int]
    KIND_FIELD_NUMBER: _ClassVar[int]
    ALIAS_FIELD_NUMBER: _ClassVar[int]
    job_id: str
    ts_unix_ms: int
    level: str
    message: str
    state: ModelState
    progress: float
    terminal: bool
    kind: JobKind
    alias: str
    def __init__(self, job_id: _Optional[str] = ..., ts_unix_ms: _Optional[int] = ..., level: _Optional[str] = ..., message: _Optional[str] = ..., state: _Optional[_Union[ModelState, str]] = ..., progress: _Optional[float] = ..., terminal: _Optional[bool] = ..., kind: _Optional[_Union[JobKind, str]] = ..., alias: _Optional[str] = ...) -> None: ...

class LoadModelRequest(_message.Message):
    __slots__ = ("alias", "force", "keep_failed", "timeout_s")
    ALIAS_FIELD_NUMBER: _ClassVar[int]
    FORCE_FIELD_NUMBER: _ClassVar[int]
    KEEP_FAILED_FIELD_NUMBER: _ClassVar[int]
    TIMEOUT_S_FIELD_NUMBER: _ClassVar[int]
    alias: str
    force: bool
    keep_failed: bool
    timeout_s: int
    def __init__(self, alias: _Optional[str] = ..., force: _Optional[bool] = ..., keep_failed: _Optional[bool] = ..., timeout_s: _Optional[int] = ...) -> None: ...

class UnloadModelRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class GetJobEventsRequest(_message.Message):
    __slots__ = ("job_id", "follow")
    JOB_ID_FIELD_NUMBER: _ClassVar[int]
    FOLLOW_FIELD_NUMBER: _ClassVar[int]
    job_id: str
    follow: bool
    def __init__(self, job_id: _Optional[str] = ..., follow: _Optional[bool] = ...) -> None: ...

class StreamLogsRequest(_message.Message):
    __slots__ = ("tail", "follow")
    TAIL_FIELD_NUMBER: _ClassVar[int]
    FOLLOW_FIELD_NUMBER: _ClassVar[int]
    tail: int
    follow: bool
    def __init__(self, tail: _Optional[int] = ..., follow: _Optional[bool] = ...) -> None: ...

class LogLine(_message.Message):
    __slots__ = ("ts_unix_ms", "line")
    TS_UNIX_MS_FIELD_NUMBER: _ClassVar[int]
    LINE_FIELD_NUMBER: _ClassVar[int]
    ts_unix_ms: int
    line: str
    def __init__(self, ts_unix_ms: _Optional[int] = ..., line: _Optional[str] = ...) -> None: ...

class GetGpuStatsRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class GpuProcess(_message.Message):
    __slots__ = ("pid", "name", "used_mib")
    PID_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    USED_MIB_FIELD_NUMBER: _ClassVar[int]
    pid: int
    name: str
    used_mib: int
    def __init__(self, pid: _Optional[int] = ..., name: _Optional[str] = ..., used_mib: _Optional[int] = ...) -> None: ...

class GpuStats(_message.Message):
    __slots__ = ("used_mib", "total_mib", "utilization_pct", "processes")
    USED_MIB_FIELD_NUMBER: _ClassVar[int]
    TOTAL_MIB_FIELD_NUMBER: _ClassVar[int]
    UTILIZATION_PCT_FIELD_NUMBER: _ClassVar[int]
    PROCESSES_FIELD_NUMBER: _ClassVar[int]
    used_mib: int
    total_mib: int
    utilization_pct: int
    processes: _containers.RepeatedCompositeFieldContainer[GpuProcess]
    def __init__(self, used_mib: _Optional[int] = ..., total_mib: _Optional[int] = ..., utilization_pct: _Optional[int] = ..., processes: _Optional[_Iterable[_Union[GpuProcess, _Mapping]]] = ...) -> None: ...

class ChatMessage(_message.Message):
    __slots__ = ("role", "content")
    ROLE_FIELD_NUMBER: _ClassVar[int]
    CONTENT_FIELD_NUMBER: _ClassVar[int]
    role: str
    content: str
    def __init__(self, role: _Optional[str] = ..., content: _Optional[str] = ...) -> None: ...

class ChatRequest(_message.Message):
    __slots__ = ("alias", "messages", "temperature", "max_tokens")
    ALIAS_FIELD_NUMBER: _ClassVar[int]
    MESSAGES_FIELD_NUMBER: _ClassVar[int]
    TEMPERATURE_FIELD_NUMBER: _ClassVar[int]
    MAX_TOKENS_FIELD_NUMBER: _ClassVar[int]
    alias: str
    messages: _containers.RepeatedCompositeFieldContainer[ChatMessage]
    temperature: float
    max_tokens: int
    def __init__(self, alias: _Optional[str] = ..., messages: _Optional[_Iterable[_Union[ChatMessage, _Mapping]]] = ..., temperature: _Optional[float] = ..., max_tokens: _Optional[int] = ...) -> None: ...

class ChatChunk(_message.Message):
    __slots__ = ("delta", "finish_reason", "prompt_tokens", "completion_tokens")
    DELTA_FIELD_NUMBER: _ClassVar[int]
    FINISH_REASON_FIELD_NUMBER: _ClassVar[int]
    PROMPT_TOKENS_FIELD_NUMBER: _ClassVar[int]
    COMPLETION_TOKENS_FIELD_NUMBER: _ClassVar[int]
    delta: str
    finish_reason: str
    prompt_tokens: int
    completion_tokens: int
    def __init__(self, delta: _Optional[str] = ..., finish_reason: _Optional[str] = ..., prompt_tokens: _Optional[int] = ..., completion_tokens: _Optional[int] = ...) -> None: ...

class ListDownloadsRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class DownloadInfo(_message.Message):
    __slots__ = ("id", "hf_repo", "include", "progress", "bytes_done", "bytes_total", "done", "ok", "error", "current_file", "revision", "started_at_unix_ms", "finished_at_unix_ms")
    ID_FIELD_NUMBER: _ClassVar[int]
    HF_REPO_FIELD_NUMBER: _ClassVar[int]
    INCLUDE_FIELD_NUMBER: _ClassVar[int]
    PROGRESS_FIELD_NUMBER: _ClassVar[int]
    BYTES_DONE_FIELD_NUMBER: _ClassVar[int]
    BYTES_TOTAL_FIELD_NUMBER: _ClassVar[int]
    DONE_FIELD_NUMBER: _ClassVar[int]
    OK_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    CURRENT_FILE_FIELD_NUMBER: _ClassVar[int]
    REVISION_FIELD_NUMBER: _ClassVar[int]
    STARTED_AT_UNIX_MS_FIELD_NUMBER: _ClassVar[int]
    FINISHED_AT_UNIX_MS_FIELD_NUMBER: _ClassVar[int]
    id: str
    hf_repo: str
    include: _containers.RepeatedScalarFieldContainer[str]
    progress: float
    bytes_done: int
    bytes_total: int
    done: bool
    ok: bool
    error: str
    current_file: str
    revision: str
    started_at_unix_ms: int
    finished_at_unix_ms: int
    def __init__(self, id: _Optional[str] = ..., hf_repo: _Optional[str] = ..., include: _Optional[_Iterable[str]] = ..., progress: _Optional[float] = ..., bytes_done: _Optional[int] = ..., bytes_total: _Optional[int] = ..., done: _Optional[bool] = ..., ok: _Optional[bool] = ..., error: _Optional[str] = ..., current_file: _Optional[str] = ..., revision: _Optional[str] = ..., started_at_unix_ms: _Optional[int] = ..., finished_at_unix_ms: _Optional[int] = ...) -> None: ...

class ListDownloadsResponse(_message.Message):
    __slots__ = ("downloads",)
    DOWNLOADS_FIELD_NUMBER: _ClassVar[int]
    downloads: _containers.RepeatedCompositeFieldContainer[DownloadInfo]
    def __init__(self, downloads: _Optional[_Iterable[_Union[DownloadInfo, _Mapping]]] = ...) -> None: ...

class DownloadModelRequest(_message.Message):
    __slots__ = ("hf_repo", "include", "revision")
    HF_REPO_FIELD_NUMBER: _ClassVar[int]
    INCLUDE_FIELD_NUMBER: _ClassVar[int]
    REVISION_FIELD_NUMBER: _ClassVar[int]
    hf_repo: str
    include: _containers.RepeatedScalarFieldContainer[str]
    revision: str
    def __init__(self, hf_repo: _Optional[str] = ..., include: _Optional[_Iterable[str]] = ..., revision: _Optional[str] = ...) -> None: ...

class DownloadProgress(_message.Message):
    __slots__ = ("id", "progress", "bytes_done", "bytes_total", "current_file", "terminal", "error", "ok")
    ID_FIELD_NUMBER: _ClassVar[int]
    PROGRESS_FIELD_NUMBER: _ClassVar[int]
    BYTES_DONE_FIELD_NUMBER: _ClassVar[int]
    BYTES_TOTAL_FIELD_NUMBER: _ClassVar[int]
    CURRENT_FILE_FIELD_NUMBER: _ClassVar[int]
    TERMINAL_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    OK_FIELD_NUMBER: _ClassVar[int]
    id: str
    progress: float
    bytes_done: int
    bytes_total: int
    current_file: str
    terminal: bool
    error: str
    ok: bool
    def __init__(self, id: _Optional[str] = ..., progress: _Optional[float] = ..., bytes_done: _Optional[int] = ..., bytes_total: _Optional[int] = ..., current_file: _Optional[str] = ..., terminal: _Optional[bool] = ..., error: _Optional[str] = ..., ok: _Optional[bool] = ...) -> None: ...

class CancelDownloadRequest(_message.Message):
    __slots__ = ("id",)
    ID_FIELD_NUMBER: _ClassVar[int]
    id: str
    def __init__(self, id: _Optional[str] = ...) -> None: ...

class CancelDownloadResponse(_message.Message):
    __slots__ = ("cancelled",)
    CANCELLED_FIELD_NUMBER: _ClassVar[int]
    cancelled: bool
    def __init__(self, cancelled: _Optional[bool] = ...) -> None: ...

class ReloadRegistryRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class ReloadRegistryResponse(_message.Message):
    __slots__ = ("errors",)
    ERRORS_FIELD_NUMBER: _ClassVar[int]
    errors: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, errors: _Optional[_Iterable[str]] = ...) -> None: ...

class RegistrationSpec(_message.Message):
    __slots__ = ("hf_repo", "alias", "quantization", "max_model_len", "gpu_memory_utilization", "host_port", "tool_parser", "use_case", "tags", "display_name", "container_name", "extra_args")
    HF_REPO_FIELD_NUMBER: _ClassVar[int]
    ALIAS_FIELD_NUMBER: _ClassVar[int]
    QUANTIZATION_FIELD_NUMBER: _ClassVar[int]
    MAX_MODEL_LEN_FIELD_NUMBER: _ClassVar[int]
    GPU_MEMORY_UTILIZATION_FIELD_NUMBER: _ClassVar[int]
    HOST_PORT_FIELD_NUMBER: _ClassVar[int]
    TOOL_PARSER_FIELD_NUMBER: _ClassVar[int]
    USE_CASE_FIELD_NUMBER: _ClassVar[int]
    TAGS_FIELD_NUMBER: _ClassVar[int]
    DISPLAY_NAME_FIELD_NUMBER: _ClassVar[int]
    CONTAINER_NAME_FIELD_NUMBER: _ClassVar[int]
    EXTRA_ARGS_FIELD_NUMBER: _ClassVar[int]
    hf_repo: str
    alias: str
    quantization: str
    max_model_len: int
    gpu_memory_utilization: float
    host_port: int
    tool_parser: str
    use_case: str
    tags: _containers.RepeatedScalarFieldContainer[str]
    display_name: str
    container_name: str
    extra_args: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, hf_repo: _Optional[str] = ..., alias: _Optional[str] = ..., quantization: _Optional[str] = ..., max_model_len: _Optional[int] = ..., gpu_memory_utilization: _Optional[float] = ..., host_port: _Optional[int] = ..., tool_parser: _Optional[str] = ..., use_case: _Optional[str] = ..., tags: _Optional[_Iterable[str]] = ..., display_name: _Optional[str] = ..., container_name: _Optional[str] = ..., extra_args: _Optional[_Iterable[str]] = ...) -> None: ...

class ProposeRegistrationRequest(_message.Message):
    __slots__ = ("spec",)
    SPEC_FIELD_NUMBER: _ClassVar[int]
    spec: RegistrationSpec
    def __init__(self, spec: _Optional[_Union[RegistrationSpec, _Mapping]] = ...) -> None: ...

class ProposeRegistrationResponse(_message.Message):
    __slots__ = ("model", "warnings", "errors")
    MODEL_FIELD_NUMBER: _ClassVar[int]
    WARNINGS_FIELD_NUMBER: _ClassVar[int]
    ERRORS_FIELD_NUMBER: _ClassVar[int]
    model: ModelInfo
    warnings: _containers.RepeatedScalarFieldContainer[str]
    errors: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, model: _Optional[_Union[ModelInfo, _Mapping]] = ..., warnings: _Optional[_Iterable[str]] = ..., errors: _Optional[_Iterable[str]] = ...) -> None: ...

class WriteRegistrationRequest(_message.Message):
    __slots__ = ("spec",)
    SPEC_FIELD_NUMBER: _ClassVar[int]
    spec: RegistrationSpec
    def __init__(self, spec: _Optional[_Union[RegistrationSpec, _Mapping]] = ...) -> None: ...

class WriteRegistrationResponse(_message.Message):
    __slots__ = ("ok", "errors")
    OK_FIELD_NUMBER: _ClassVar[int]
    ERRORS_FIELD_NUMBER: _ClassVar[int]
    ok: bool
    errors: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, ok: _Optional[bool] = ..., errors: _Optional[_Iterable[str]] = ...) -> None: ...

class ListCacheRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class CacheEntry(_message.Message):
    __slots__ = ("repo_id", "size_bytes", "last_modified", "aliases")
    REPO_ID_FIELD_NUMBER: _ClassVar[int]
    SIZE_BYTES_FIELD_NUMBER: _ClassVar[int]
    LAST_MODIFIED_FIELD_NUMBER: _ClassVar[int]
    ALIASES_FIELD_NUMBER: _ClassVar[int]
    repo_id: str
    size_bytes: int
    last_modified: str
    aliases: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, repo_id: _Optional[str] = ..., size_bytes: _Optional[int] = ..., last_modified: _Optional[str] = ..., aliases: _Optional[_Iterable[str]] = ...) -> None: ...

class ListCacheResponse(_message.Message):
    __slots__ = ("entries", "disk_total_bytes", "disk_free_bytes")
    ENTRIES_FIELD_NUMBER: _ClassVar[int]
    DISK_TOTAL_BYTES_FIELD_NUMBER: _ClassVar[int]
    DISK_FREE_BYTES_FIELD_NUMBER: _ClassVar[int]
    entries: _containers.RepeatedCompositeFieldContainer[CacheEntry]
    disk_total_bytes: int
    disk_free_bytes: int
    def __init__(self, entries: _Optional[_Iterable[_Union[CacheEntry, _Mapping]]] = ..., disk_total_bytes: _Optional[int] = ..., disk_free_bytes: _Optional[int] = ...) -> None: ...

class RemoveCachedRequest(_message.Message):
    __slots__ = ("repo_id",)
    REPO_ID_FIELD_NUMBER: _ClassVar[int]
    repo_id: str
    def __init__(self, repo_id: _Optional[str] = ...) -> None: ...

class RemoveCachedResponse(_message.Message):
    __slots__ = ("removed", "error")
    REMOVED_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    removed: bool
    error: str
    def __init__(self, removed: _Optional[bool] = ..., error: _Optional[str] = ...) -> None: ...
