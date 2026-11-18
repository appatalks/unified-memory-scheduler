"""User-selected model-storage locations and mount checks."""

import os
from pathlib import Path
import re
import stat

DEVICE_UUID = re.compile(r"[A-Za-z0-9][A-Za-z0-9-]{0,127}")
_mount = os.environ.get("VRAMBRIDGE_MODEL_MOUNT")
_uuid = os.environ.get("VRAMBRIDGE_MODEL_UUID", "")
_library = os.environ.get("VRAMBRIDGE_MODEL_LIBRARY")
NVME_MOUNT = Path(_mount) if _mount else None
NVME_UUID_DEVICE = Path("/dev/disk/by-uuid") / _uuid if DEVICE_UUID.fullmatch(_uuid) else None
NVME_LIBRARY_ROOT = Path(_library) if _library else NVME_MOUNT
NVME_MODELS_ROOT = NVME_LIBRARY_ROOT / "models" if NVME_LIBRARY_ROOT is not None else None
NVME_STAGING_ROOT = NVME_LIBRARY_ROOT / "model-staging" if NVME_LIBRARY_ROOT is not None else None


def check_selected_mount(path: Path, mount: Path | None, require_mount) -> None:
    if mount is not None and (Path(os.path.abspath(path)).is_relative_to(mount) or
                              path.resolve(strict=False).is_relative_to(mount)):
        require_mount()


def require_nvme_mount() -> None:
    if (NVME_MOUNT is None or NVME_UUID_DEVICE is None or NVME_LIBRARY_ROOT is None or
            NVME_MODELS_ROOT is None or not NVME_MOUNT.is_absolute() or not NVME_LIBRARY_ROOT.is_absolute()):
        raise ValueError("Set VRAMBRIDGE_MODEL_MOUNT and a valid VRAMBRIDGE_MODEL_UUID; "
                         "VRAMBRIDGE_MODEL_LIBRARY, if set, must be an absolute path on that mount.")
    try:
        device = NVME_UUID_DEVICE.stat()
        mount = NVME_MOUNT.resolve(strict=True)
        library = NVME_LIBRARY_ROOT.resolve(strict=True)
        models = NVME_MODELS_ROOT.resolve(strict=True)
        if (not NVME_MOUNT.is_mount() or not stat.S_ISBLK(device.st_mode) or
                NVME_MOUNT.stat().st_dev != device.st_rdev or not NVME_MODELS_ROOT.is_dir() or
                not library.is_relative_to(mount) or library.stat().st_dev != device.st_rdev or
                not models.is_relative_to(library) or models.stat().st_dev != device.st_rdev):
            raise ValueError("The selected NVMe model volume is not mounted at its expected path.")
    except OSError as exc:
        raise ValueError("The selected NVMe model volume is not mounted at its expected path.") from exc
