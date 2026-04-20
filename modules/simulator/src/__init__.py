
from __future__ import annotations
import errno
import logging
import os
import sys
import tempfile
from distutils.util import strtobool
from pathlib import Path

# --------------------------------------------------------------------------- #
# Public package metadata                                                     #
# --------------------------------------------------------------------------- #
__version__ = "0.1.0"          # update as needed
__author__  = "Shirley Wu & the CollabLLM team"

__all__ = [
    "__version__",
    "ENABLE_COLLABLLM_LOGGING",
    "RUN_USER_DIR",
]

# --------------------------------------------------------------------------- #
# Utility: boolean env-var parser                                             #
# --------------------------------------------------------------------------- #
def _env_flag(name: str, default: str = "1") -> bool:

    try:
        return bool(strtobool(os.getenv(name, default)))
    except ValueError:
        # Invalid value; fall back to default.
        return bool(strtobool(default))


# --------------------------------------------------------------------------- #
# Global logging switch                                                       #
# --------------------------------------------------------------------------- #
ENABLE_COLLABLLM_LOGGING: bool = _env_flag("ENABLE_COLLABLLM_LOGGING", "1")


_pkg_logger = logging.getLogger("collabllm")

if ENABLE_COLLABLLM_LOGGING:
    # Configure basic console output if the user hasn’t configured logging yet.
    # We guard with "if not root.handlers" to avoid double-configuration.
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )
    _pkg_logger.info("CollabLLM logging enabled.")
else:
    # Silence *all* log records emitted from collabllm.* by:
    # 1) setting a level higher than CRITICAL
    # 2) preventing propagation to the root logger
    # 3) attaching a NullHandler
    _pkg_logger.setLevel(logging.CRITICAL)
    _pkg_logger.propagate = False
    _pkg_logger.handlers.clear()
    _pkg_logger.addHandler(logging.NullHandler())


# --------------------------------------------------------------------------- #
# LiteLLM                                                                     #
# --------------------------------------------------------------------------- #
import litellm

# Diable the cache by default, as it is not needed in most cases.
litellm.disable_cache()

# Also silence the LiteLLM logger.
logging.getLogger("LiteLLM").setLevel(logging.CRITICAL)
_pkg_logger.info("Disable LiteLLM cache and logging by default. ")

# --------------------------------------------------------------------------- #
# Per-user runtime directory                                                  #
# --------------------------------------------------------------------------- #
_DEFAULT_RUN_DIR = "/run/user/{uid}/collabllm"


def _platform_default_run_dir() -> Path:
    xdg_runtime_dir = os.getenv("XDG_RUNTIME_DIR")
    if xdg_runtime_dir:
        return Path(xdg_runtime_dir).expanduser() / "collabllm"

    if sys.platform.startswith("linux"):
        return Path(_DEFAULT_RUN_DIR.format(uid=os.getuid()))

    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches" / "collabllm"

    return Path(tempfile.gettempdir()) / "collabllm"

def _resolve_run_user_dir() -> Path:
    # 1) honour explicit env-var
    env_val = os.getenv("RUN_USER_DIR")
    if env_val:
        return Path(env_val).expanduser()

    # 2) fall back to a platform-appropriate runtime/cache path
    return _platform_default_run_dir()

RUN_USER_DIR: Path = _resolve_run_user_dir()
os.environ["RUN_USER_DIR"] = str(RUN_USER_DIR) 

try:
    RUN_USER_DIR.mkdir(parents=True, exist_ok=True)
except OSError as exc:
    if exc.errno in {errno.EACCES, errno.ENOENT, errno.EROFS}:
        if sys.platform == "darwin":
            fallback = Path.home() / "Library" / "Caches" / "collabllm"
        else:
            fallback = Path.home() / ".cache" / "collabllm"
        fallback.mkdir(parents=True, exist_ok=True)
        _pkg_logger.warning(
            "Cannot access %s; using %s instead.", RUN_USER_DIR, fallback
        )
        RUN_USER_DIR = fallback
        os.environ["RUN_USER_DIR"] = str(RUN_USER_DIR)
    else:
        raise





