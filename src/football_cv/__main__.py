"""Support `python -m football_cv` with the same exit codes as the CLI command."""
from .app import main
raise SystemExit(main())
