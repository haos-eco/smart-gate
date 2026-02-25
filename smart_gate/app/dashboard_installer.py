import os
import shutil
from constants import DASHBOARD_SRC, DASHBOARD_DST


def install_dashboard() -> None:
    """
    Copy bundled dashboard files from DASHBOARD_SRC to DASHBOARD_DST.
    If the destination already exists it is fully removed and recreated,
    so any addon update is always reflected without manual intervention.
    """
    if not os.path.isdir(DASHBOARD_SRC):
        print(
            f"[SmartGate] Dashboard source not found at {DASHBOARD_SRC} — skipping install"
        )
        return

    # Remove stale installation
    if os.path.exists(DASHBOARD_DST):
        shutil.rmtree(DASHBOARD_DST)
        print(f"[SmartGate] Removed existing dashboard at {DASHBOARD_DST}")

    shutil.copytree(DASHBOARD_SRC, DASHBOARD_DST)
    print(f"[SmartGate] Dashboard installed: {DASHBOARD_SRC} → {DASHBOARD_DST}")
