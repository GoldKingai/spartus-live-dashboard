"""Auto-update system for Spartus Live Dashboard.

Checks GitHub releases for new versions, downloads and applies updates
using git pull. Supports both git-cloned repos and standalone installs.

Flow:
    1. On startup, check GitHub API for latest release tag
    2. Compare with local version (from pyproject.toml)
    3. If newer version exists, notify user via callback
    4. On user confirmation, pull updates and signal restart

Non-blocking: version check runs in a background thread.
"""

import json
import os
import subprocess
import sys
import threading
import urllib.request
import urllib.error
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional


# --- Version parsing ---------------------------------------------------------

def parse_version(version_str: str) -> tuple:
    """Parse 'X.Y.Z' into (X, Y, Z) tuple for comparison."""
    parts = version_str.strip().lstrip("v").split(".")
    return tuple(int(p) for p in parts[:3])


def get_local_version(dashboard_root: Path = None) -> str:
    """Read version from pyproject.toml."""
    if dashboard_root is None:
        dashboard_root = Path(__file__).resolve().parent.parent

    pyproject = dashboard_root / "pyproject.toml"
    if not pyproject.exists():
        return "0.0.0"

    with open(pyproject, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip().startswith("version"):
                # version = "1.0.0"
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return "0.0.0"


# --- Data classes ------------------------------------------------------------

@dataclass
class UpdateInfo:
    """Information about an available update."""
    current_version: str
    latest_version: str
    release_notes: str
    release_url: str
    published_at: str
    is_newer: bool


# --- Auto Updater ------------------------------------------------------------

class AutoUpdater:
    """Manages checking for and applying updates from GitHub.

    Args:
        repo_owner: GitHub username/org (e.g. "GoldKingai")
        repo_name: GitHub repo name (e.g. "spartus-live-dashboard")
        dashboard_root: Path to the live_dashboard directory
        on_update_available: Callback(UpdateInfo) when update found
        on_update_progress: Callback(message: str) for progress updates
        on_update_complete: Callback(success: bool, message: str) when done
    """

    GITHUB_API = "https://api.github.com/repos/{owner}/{repo}/releases/latest"
    CHECK_TIMEOUT = 10  # seconds

    def __init__(
        self,
        repo_owner: str = "GoldKingai",
        repo_name: str = "spartus-live-dashboard",
        dashboard_root: Path = None,
        on_update_available: Optional[Callable] = None,
        on_update_progress: Optional[Callable] = None,
        on_update_complete: Optional[Callable] = None,
        on_no_update: Optional[Callable] = None,
        on_check_failed: Optional[Callable] = None,
    ):
        self.repo_owner = repo_owner
        self.repo_name = repo_name
        self.root = Path(dashboard_root or Path(__file__).resolve().parent.parent)
        self.on_update_available = on_update_available
        self.on_update_progress = on_update_progress
        self.on_update_complete = on_update_complete
        self.on_no_update = on_no_update
        self.on_check_failed = on_check_failed

        self._current_version = get_local_version(self.root)
        self._latest_info: Optional[UpdateInfo] = None
        self._checking = False

    @property
    def current_version(self) -> str:
        return self._current_version

    @property
    def latest_info(self) -> Optional[UpdateInfo]:
        return self._latest_info

    # --- Check for updates (background) --------------------------------------

    def check_for_updates(self, blocking: bool = False):
        """Check GitHub for a newer release.

        Args:
            blocking: If True, run synchronously. If False, run in background thread.
        """
        if self._checking:
            return

        if blocking:
            self._do_check()
        else:
            t = threading.Thread(target=self._do_check, daemon=True)
            t.start()

    def _do_check(self):
        """Perform the version check against GitHub API."""
        self._checking = True
        try:
            url = self.GITHUB_API.format(owner=self.repo_owner, repo=self.repo_name)
            req = urllib.request.Request(
                url,
                headers={"Accept": "application/vnd.github.v3+json",
                          "User-Agent": f"SpartusLiveDashboard/{self._current_version}"},
            )

            with urllib.request.urlopen(req, timeout=self.CHECK_TIMEOUT) as resp:
                data = json.loads(resp.read().decode("utf-8"))

            latest_tag = data.get("tag_name", "0.0.0")
            latest_version = latest_tag.lstrip("v")
            release_notes = data.get("body", "No release notes.")
            release_url = data.get("html_url", "")
            published_at = data.get("published_at", "")

            is_newer = parse_version(latest_version) > parse_version(self._current_version)

            self._latest_info = UpdateInfo(
                current_version=self._current_version,
                latest_version=latest_version,
                release_notes=release_notes,
                release_url=release_url,
                published_at=published_at,
                is_newer=is_newer,
            )

            if is_newer and self.on_update_available:
                self.on_update_available(self._latest_info)
            elif not is_newer and self.on_no_update:
                self.on_no_update()

        except urllib.error.URLError:
            if self.on_check_failed:
                self.on_check_failed()
        except Exception:
            if self.on_check_failed:
                self.on_check_failed()
        finally:
            self._checking = False

    # --- Apply update ---------------------------------------------------------

    def apply_update(self) -> bool:
        """Pull the latest changes from GitHub.

        Uses git pull if the dashboard is a git repo.
        Falls back to a manual download prompt if not.

        Returns:
            True if update was applied and restart is needed.
        """
        try:
            if self._is_git_repo():
                return self._update_via_git()
            else:
                return self._update_via_download()
        except Exception as e:
            if self.on_update_complete:
                self.on_update_complete(False, f"Update failed: {e}")
            return False

    def _is_git_repo(self) -> bool:
        """Check if the dashboard directory is a git repository."""
        return (self.root / ".git").exists()

    def _update_via_git(self) -> bool:
        """Update by running git pull."""
        self._progress("Checking for uncommitted changes...")

        # Stash any local changes
        result = self._run_git("stash", "--include-untracked")
        had_stash = "No local changes" not in (result or "")

        self._progress("Pulling latest version from GitHub...")

        # Pull latest
        result = self._run_git("pull", "origin", "master")
        if result is None:
            if had_stash:
                self._run_git("stash", "pop")
            if self.on_update_complete:
                self.on_update_complete(False, "Git pull failed")
            return False

        # Check if requirements changed
        if "requirements.txt" in (result or ""):
            self._progress("Installing updated dependencies...")
            self._install_requirements()

        # Pop stash if we had local changes
        if had_stash:
            self._progress("Restoring local configuration...")
            self._run_git("stash", "pop")

        # Update local version cache
        self._current_version = get_local_version(self.root)

        self._progress("Update complete! Restart to apply changes.")
        if self.on_update_complete:
            self.on_update_complete(True, f"Updated to v{self._current_version}")
        return True

    def _update_via_download(self) -> bool:
        """For non-git installs, prompt user to download manually."""
        if self._latest_info and self.on_update_complete:
            self.on_update_complete(
                False,
                f"Manual download required.\n"
                f"Download v{self._latest_info.latest_version} from:\n"
                f"{self._latest_info.release_url}"
            )
        return False

    def _run_git(self, *args) -> Optional[str]:
        """Run a git command in the dashboard directory."""
        try:
            result = subprocess.run(
                ["git"] + list(args),
                cwd=str(self.root),
                capture_output=True,
                text=True,
                timeout=60,
            )
            if result.returncode == 0:
                return result.stdout
            return None
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return None

    def _install_requirements(self):
        """Install/update requirements after a pull."""
        req_file = self.root / "requirements.txt"
        if not req_file.exists():
            return

        python = sys.executable
        try:
            subprocess.run(
                [python, "-m", "pip", "install", "-r", str(req_file), "-q"],
                cwd=str(self.root),
                capture_output=True,
                timeout=120,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

    def _progress(self, message: str):
        """Send a progress update."""
        if self.on_update_progress:
            self.on_update_progress(message)
