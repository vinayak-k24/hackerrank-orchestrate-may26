import os
from pathlib import Path

def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value

def bootstrap_environment(repo_root: Path) -> None:
    load_env_file(repo_root / ".env")
    load_env_file(repo_root / ".env.local")

def get_config() -> dict[str, str]:
    return {
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY", ""),
        "OPENAI_MODEL": os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
        "OPENAI_TEMPERATURE": os.getenv("OPENAI_TEMPERATURE", "0.3"),
        "OPENAI_MAX_TOKENS": os.getenv("OPENAI_MAX_TOKENS", "250"),
        "USE_OPENAI": os.getenv("USE_OPENAI", "false").lower(),
    }
