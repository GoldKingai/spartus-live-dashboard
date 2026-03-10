"""Auto-update system for Spartus Live Dashboard.

Checks GitHub releases for new versions, downloads and applies updates.
Supports both git-cloned repos (git pull) and ZIP-download installs
(downloads source ZIP, extracts, replaces files while preserving user data).

Flow:
    1. On startup, check GitHub API for latest release tag
    2. Compare with local version (from pyproject.toml)
    3. If newer version exists, notify user via callback
    4. On user confirmation:
       a. Git repos:  git stash → git pull → git stash pop
       b. ZIP installs: download ZIP → extract → replace files → cleanup

Non-blocking: version check and update both run in background threads.

User data preserved during ZIP update:
    - model/          (user's deployed model files)
    - storage/        (logs, memory DB, state, screenshots, reports)
    - config/default_config.yaml  (user's custom configuration)
"""

import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import urllib.request
import urllib.error
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional

log = logging.getLogger(__name__)

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
    zipball_url: str = ""


# Directories and files to preserve during ZIP update (never overwritten)
_PRESERVE_DIRS = ["model", "storage"]
_PRESERVE_FILES = [os.path.join("config", "default_config.yaml")]


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
        on_no_update: Callback() when already on latest
        on_check_failed: Callback() when check fails
    """

    GITHUB_API = "https://api.github.com/repos/{owner}/{repo}/releases/latest"
    GITHUB_ZIP = "https://github.com/{owner}/{repo}/archive/refs/tags/{tag}.zip"
    CHECK_TIMEOUT = 10  # seconds
    DOWNLOAD_TIMEOUT = 120  # seconds

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
            zipball_url = data.get("zipball_url", "")

            is_newer = parse_version(latest_version) > parse_version(self._current_version)

            self._latest_info = UpdateInfo(
                current_version=self._current_version,
                latest_version=latest_version,
                release_notes=release_notes,
                release_url=release_url,
                published_at=published_at,
                is_newer=is_newer,
                zipball_url=zipball_url,
            )

            if is_newer and self.on_update_available:
                self.on_update_available(self._latest_info)
            elif not is_newer and self.on_no_update:
                self.on_no_update()

        except urllib.error.URLError as e:
            log.warning("Update check failed (network): %s", e)
            if self.on_check_failed:
                self.on_check_failed()
        except Exception as e:
            log.warning("Update check failed: %s", e)
            if self.on_check_failed:
                self.on_check_failed()
        finally:
            self._checking = False

    # --- Apply update ---------------------------------------------------------

    def apply_update(self) -> bool:
        """Apply the latest update.

        Uses git pull if the dashboard is a git repo.
        Falls back to ZIP download + file replacement if not.

        Returns:
            True if update was applied and restart is needed.
        """
        try:
            if self._is_git_repo():
                return self._update_via_git()
            else:
                return self._update_via_zip()
        except Exception as e:
            log.error("Update failed: %s", e, exc_info=True)
            if self.on_update_complete:
                self.on_update_complete(False, f"Update failed: {e}")
            return False

    def _is_git_repo(self) -> bool:
        """Check if the dashboard directory is a git repository."""
        return (self.root / ".git").exists()

    # --- Git-based update -----------------------------------------------------

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

    # --- ZIP-based update (for non-git installs) ------------------------------

    def _update_via_zip(self) -> bool:
        """Update by downloading source ZIP from GitHub and replacing files.

        Steps:
            1. Download source ZIP from GitHub tags
            2. Extract to temp directory
            3. Back up current installation
            4. Copy new files, preserving user data dirs
            5. Install updated requirements if changed
            6. Clean up temp files
        """
        if not self._latest_info:
            if self.on_update_complete:
                self.on_update_complete(False, "No update info available")
            return False

        tag = f"v{self._latest_info.latest_version}"
        zip_url = self.GITHUB_ZIP.format(
            owner=self.repo_owner,
            repo=self.repo_name,
            tag=tag,
        )

        tmp_dir = None
        backup_dir = None

        try:
            # --- Step 1: Download ZIP ---
            self._progress(f"Downloading v{self._latest_info.latest_version}...")
            log.info("Downloading update from: %s", zip_url)

            tmp_dir = Path(tempfile.mkdtemp(prefix="spartus_update_"))
            zip_path = tmp_dir / "update.zip"

            req = urllib.request.Request(
                zip_url,
                headers={"User-Agent": f"SpartusLiveDashboard/{self._current_version}"},
            )
            with urllib.request.urlopen(req, timeout=self.DOWNLOAD_TIMEOUT) as resp:
                total = int(resp.headers.get("Content-Length", 0))
                downloaded = 0
                chunk_size = 64 * 1024  # 64 KB chunks

                with open(zip_path, "wb") as f:
                    while True:
                        chunk = resp.read(chunk_size)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total > 0:
                            pct = int(downloaded / total * 100)
                            self._progress(
                                f"Downloading... {downloaded // 1024} KB"
                                f" / {total // 1024} KB ({pct}%)"
                            )

            log.info("Download complete: %d bytes", zip_path.stat().st_size)

            # --- Step 2: Extract ZIP ---
            self._progress("Extracting update...")

            extract_dir = tmp_dir / "extracted"
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(extract_dir)

            # GitHub ZIPs have a top-level folder like "repo-name-tag/"
            # Find it
            source_dir = self._find_source_root(extract_dir)
            if source_dir is None:
                if self.on_update_complete:
                    self.on_update_complete(False, "Invalid ZIP structure")
                return False

            log.info("Source root in ZIP: %s", source_dir)

            # --- Step 3: Back up current installation ---
            self._progress("Backing up current installation...")

            backup_dir = Path(tempfile.mkdtemp(prefix="spartus_backup_"))
            files_backed_up = self._backup_current(backup_dir)
            log.info("Backed up %d items to %s", files_backed_up, backup_dir)

            # --- Step 4: Replace files ---
            self._progress("Installing new files...")

            files_updated = self._replace_files(source_dir)
            log.info("Updated %d files", files_updated)

            # --- Step 5: Install requirements if changed ---
            new_req = self.root / "requirements.txt"
            if new_req.exists():
                self._progress("Checking dependencies...")
                self._install_requirements()

            # --- Step 6: Verify ---
            new_version = get_local_version(self.root)
            self._current_version = new_version

            self._progress("Update complete! Restart to apply changes.")
            log.info("Update applied: v%s", new_version)

            if self.on_update_complete:
                self.on_update_complete(True, f"Updated to v{new_version}")
            return True

        except urllib.error.URLError as e:
            log.error("Download failed: %s", e)
            self._rollback(backup_dir)
            if self.on_update_complete:
                self.on_update_complete(False, f"Download failed: {e}")
            return False

        except Exception as e:
            log.error("ZIP update failed: %s", e, exc_info=True)
            self._rollback(backup_dir)
            if self.on_update_complete:
                self.on_update_complete(False, f"Update failed: {e}")
            return False

        finally:
            # Clean up temp files (but keep backup for safety)
            if tmp_dir and tmp_dir.exists():
                try:
                    shutil.rmtree(tmp_dir, ignore_errors=True)
                except Exception:
                    pass

    def _find_source_root(self, extract_dir: Path) -> Optional[Path]:
        """Find the actual source root inside the extracted ZIP.

        GitHub source ZIPs contain a single top-level directory like
        'spartus-live-dashboard-1.2.0/' that holds all the files.
        """
        entries = list(extract_dir.iterdir())
        if len(entries) == 1 and entries[0].is_dir():
            candidate = entries[0]
            # Verify it looks like our project (has pyproject.toml or main.py)
            if (candidate / "pyproject.toml").exists() or (candidate / "main.py").exists():
                return candidate
        # Fallback: maybe files are directly in extract_dir
        if (extract_dir / "pyproject.toml").exists() or (extract_dir / "main.py").exists():
            return extract_dir
        return None

    def _backup_current(self, backup_dir: Path) -> int:
        """Back up current files (excluding preserved dirs) to backup_dir."""
        count = 0
        for item in self.root.iterdir():
            name = item.name
            # Skip preserved directories, venv, __pycache__, .git
            if name in _PRESERVE_DIRS or name in ("venv", ".venv", "__pycache__", ".git"):
                continue
            dest = backup_dir / name
            try:
                if item.is_dir():
                    shutil.copytree(item, dest, dirs_exist_ok=True)
                else:
                    shutil.copy2(item, dest)
                count += 1
            except Exception as e:
                log.warning("Backup skipped %s: %s", name, e)
        return count

    def _replace_files(self, source_dir: Path) -> int:
        """Replace dashboard files from the new source, preserving user data.

        Rules:
            - Directories in _PRESERVE_DIRS are never touched
            - Files in _PRESERVE_FILES are never overwritten
            - venv/ is never touched
            - Everything else is replaced with the new version
        """
        count = 0
        preserve_dirs_set = set(_PRESERVE_DIRS + ["venv", ".venv", "__pycache__", ".git"])

        for item in source_dir.iterdir():
            name = item.name
            dest = self.root / name

            # Never overwrite preserved directories
            if name in preserve_dirs_set:
                continue

            # Check if this is a preserved file
            rel = name  # top-level file
            if rel in _PRESERVE_FILES:
                continue

            try:
                if item.is_dir():
                    # For directories, merge: delete old, copy new
                    # But check for preserved files inside
                    self._merge_directory(item, dest)
                else:
                    shutil.copy2(item, dest)
                count += 1
            except Exception as e:
                log.warning("Failed to update %s: %s", name, e)

        return count

    def _merge_directory(self, src_dir: Path, dest_dir: Path) -> None:
        """Merge a source directory into dest, respecting preserved files."""
        # Ensure dest exists
        dest_dir.mkdir(parents=True, exist_ok=True)

        # Track what's in the new version
        new_items = set()

        for item in src_dir.iterdir():
            new_items.add(item.name)
            dest_item = dest_dir / item.name

            # Check if this path is preserved
            try:
                rel_path = str(dest_item.relative_to(self.root)).replace("\\", "/")
            except ValueError:
                rel_path = ""

            # Normalize preserved file paths for comparison
            preserved = False
            for pf in _PRESERVE_FILES:
                if rel_path == pf.replace("\\", "/"):
                    preserved = True
                    break

            if preserved:
                continue

            if item.is_dir():
                self._merge_directory(item, dest_item)
            else:
                shutil.copy2(item, dest_item)

        # Remove files in dest that aren't in the new version
        # (but only files, not directories -- to be safe)
        if dest_dir.exists():
            for existing in dest_dir.iterdir():
                if existing.name not in new_items and existing.is_file():
                    try:
                        rel_path = str(existing.relative_to(self.root)).replace("\\", "/")
                        preserved = any(
                            rel_path == pf.replace("\\", "/") for pf in _PRESERVE_FILES
                        )
                        if not preserved:
                            existing.unlink()
                    except Exception:
                        pass

    def _rollback(self, backup_dir: Optional[Path]) -> None:
        """Attempt to restore from backup if update failed."""
        if backup_dir is None or not backup_dir.exists():
            log.warning("No backup available for rollback")
            return

        log.info("Rolling back to backup at %s", backup_dir)
        self._progress("Update failed — rolling back...")

        try:
            for item in backup_dir.iterdir():
                dest = self.root / item.name
                if item.is_dir():
                    if dest.exists():
                        shutil.rmtree(dest, ignore_errors=True)
                    shutil.copytree(item, dest, dirs_exist_ok=True)
                else:
                    shutil.copy2(item, dest)
            log.info("Rollback complete")
        except Exception as e:
            log.error("Rollback failed: %s", e)

    # --- Shared helpers -------------------------------------------------------

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
        """Install/update requirements after an update."""
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
        log.info("Update progress: %s", message)
        if self.on_update_progress:
            self.on_update_progress(message)
