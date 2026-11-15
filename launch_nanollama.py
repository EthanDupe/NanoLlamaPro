"""
launch_nanollama.py — Single-File GitHub Release Bootstrapper & Auto-Trainer
==========================================================================
Drop this single file anywhere and double-click it (zero dependencies needed):
  1. Checks your GitHub Releases (/releases/latest) for new updates.
  2. Downloads all release assets (and auto-extracts .zip archives) with progress bars.
  3. Installs required Python packages (numpy, and optionally torch) automatically.
  4. Launches self_train.py --forever (with automatic Wi-Fi/LAN multi-PC clustering).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import ssl
import subprocess
import sys
import time
import traceback
import urllib.error
import urllib.request
import zipfile

# Set this to your GitHub "username/repo" before sharing this file:
DEFAULT_GITHUB_REPO = "EthanDupe/NanoLlamaPro"

APP_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(APP_DIR, "launcher_config.json")
USER_AGENT = "NanoLlamaPro-Launcher/1.0"


def print_bar(current: int, total: int, label: str, extra: str = "", start_time: float | None = None) -> None:
    total = max(1, total)
    pct = min(1.0, current / total)
    cols = shutil.get_terminal_size((80, 24)).columns
    max_w = max(50, cols - 1)
    bar_w = 20
    filled = int(bar_w * pct)
    bar = "#" * filled + "-" * (bar_w - filled)
    eta = ""
    if start_time is not None and current > 0 and current < total:
        rem = (time.monotonic() - start_time) * (total - current) / current
        eta = f" | ETA {int(rem)}s"
    msg = f"{label:<14} [{bar}] {pct * 100:5.1f}%{eta}"
    if extra:
        msg += f" | {extra}"
    msg = msg[: max_w - 3] + "..." if len(msg) > max_w else msg.ljust(max_w)
    sys.stdout.write("\r" + msg)
    sys.stdout.flush()
    if current >= total:
        sys.stdout.write("\n")
        sys.stdout.flush()


def _urlopen(url: str, timeout: int = 20):
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/vnd.github+json"})
    try:
        return urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.URLError as exc:
        if isinstance(exc.reason, ssl.SSLError):
            ctx = ssl._create_unverified_context()
            return urllib.request.urlopen(req, timeout=timeout, context=ctx)
        raise


def download_file(url: str, dest_path: str, label: str = "Downloading") -> None:
    t0 = time.monotonic()
    with _urlopen(url, timeout=60) as resp:
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        tmp_path = dest_path + ".part"
        with open(tmp_path, "wb") as out:
            while True:
                chunk = resp.read(65536)
                if not chunk:
                    break
                out.write(chunk)
                done += len(chunk)
                if total > 0:
                    mb_d, mb_t = done / 1_048_576, total / 1_048_576
                    print_bar(done, total, label, f"{mb_d:.1f}/{mb_t:.1f} MB", start_time=t0)
                else:
                    print_bar(1, 2, label, f"{done / 1_048_576:.1f} MB")
        print_bar(1, 1, label, os.path.basename(dest_path), start_time=t0)
        os.replace(tmp_path, dest_path)


def fetch_github_release(repo: str) -> dict | None:
    url = f"https://api.github.com/repos/{repo}/releases/latest"
    try:
        with _urlopen(url, timeout=15) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        raise


def ensure_packages(install_torch: bool = True) -> None:
    print_bar(0, 2, "Dependencies", "checking numpy...")
    try:
        import numpy  # noqa: F401
    except ModuleNotFoundError:
        print("\nInstalling required package: numpy...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-warn-script-location", "numpy"])
    print_bar(1, 2, "Dependencies", "numpy OK")

    if install_torch:
        try:
            import torch  # noqa: F401
        except ModuleNotFoundError:
            print("\nInstalling PyTorch (torch) for faster batched/GPU training...", flush=True)
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-warn-script-location", "torch"])
            except Exception as exc:
                print(f"Skipping torch ({exc}); will use numpy-cpu.", flush=True)
        except Exception as exc:
            print(f"\nPyTorch import warning ({exc}); continuing with numpy-cpu.", flush=True)
    print_bar(2, 2, "Dependencies", "ready")


def sync_from_github(repo: str, force: bool = False) -> None:
    os.makedirs(APP_DIR, exist_ok=True)
    cfg = {}
    if os.path.isfile(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                cfg = json.load(f)
        except Exception:
            cfg = {}

    print(f"Checking GitHub Releases for '{repo}'...", flush=True)
    release = None
    try:
        release = fetch_github_release(repo)
    except Exception as exc:
        print(f"Could not reach GitHub ({exc}). Using local files if present.")

    if release:
        tag = release.get("tag_name", "latest")
        assets = release.get("assets", [])
        if not force and cfg.get("tag") == tag and os.path.isfile(os.path.join(APP_DIR, "self_train.py")):
            print(f"Already up to date ({tag}).", flush=True)
            return

        if assets:
            print(f"Found release {tag} ({len(assets)} file(s))!", flush=True)
            has_zip = any(a["name"].lower().endswith(".zip") for a in assets)
            for asset in assets:
                name = asset["name"]
                if has_zip and name == "launch_nanollama.py":
                    continue
                dl_url = asset["browser_download_url"]
                dest = os.path.join(APP_DIR, name)
                download_file(dl_url, dest, label=f"Get {name[:9]}")
                if name.lower().endswith(".zip"):
                    print(f"Extracting {name}...", flush=True)
                    with zipfile.ZipFile(dest, "r") as zf:
                        zf.extractall(APP_DIR)
            cfg["tag"] = tag
            cfg["repo"] = repo
            with open(CONFIG_PATH, "w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=2)
            return

    # Fallback: pull raw files from main branch if no release assets exist
    if not os.path.isfile(os.path.join(APP_DIR, "self_train.py")) or force:
        raw_base = f"https://raw.githubusercontent.com/{repo}/main"
        for fname in ("nanollama_pro.py", "self_train.py", "training_data.txt"):
            dest = os.path.join(APP_DIR, fname)
            try:
                download_file(f"{raw_base}/{fname}", dest, label=f"Get {fname[:9]}")
            except Exception as exc:
                if fname != "training_data.txt":
                    raise RuntimeError(f"Failed to download {fname} from {repo}: {exc}") from exc


def run() -> None:
    print("=" * 60, flush=True)
    print("  NanoLlamaPro — GitHub Auto-Installer & Trainer", flush=True)
    print("=" * 60, flush=True)

    parser = argparse.ArgumentParser(description="Download NanoLlamaPro from GitHub Releases and start training.")
    parser.add_argument("--repo", default=None, help="GitHub repository (owner/repo).")
    parser.add_argument("--update", action="store_true", help="Force re-download from GitHub.")
    parser.add_argument("--no-torch", action="store_true", help="Skip installing PyTorch and use numpy-cpu.")
    args, extra_args = parser.parse_known_args()

    repo = args.repo or DEFAULT_GITHUB_REPO
    if os.path.isfile(CONFIG_PATH) and not args.repo:
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                repo = json.load(f).get("repo", repo)
        except Exception:
            pass

    sync_from_github(repo, force=args.update)
    ensure_packages(install_torch=not args.no_torch)

    trainer = os.path.join(APP_DIR, "self_train.py")
    if not os.path.isfile(trainer):
        raise FileNotFoundError(
            f"Could not find self_train.py in {APP_DIR}. Check that '{repo}' has a published GitHub Release."
        )

    print("\nLaunching NanoLlamaPro Self-Training...", flush=True)
    cmd = [sys.executable, trainer] + (extra_args if extra_args else ["--forever"])
    subprocess.run(cmd, cwd=APP_DIR, check=False)


def main() -> None:
    try:
        run()
    except SystemExit:
        raise
    except Exception:
        print("\n--- LAUNCHER ERROR ---")
        traceback.print_exc()
        try:
            input("\nPress Enter to exit...")
        except (EOFError, KeyboardInterrupt):
            pass
        sys.exit(1)


if __name__ == "__main__":
    main()
