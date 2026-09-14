"""Process exit codes (FR-00.13)."""

EXIT_OK = 0
EXIT_RUNTIME = 1  # Docker/GPU/timeout failure at runtime
EXIT_CONFIG = 2  # config/validation error
EXIT_ALIAS = 3  # unknown alias
