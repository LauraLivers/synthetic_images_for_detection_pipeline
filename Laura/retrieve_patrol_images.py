#!/usr/bin/env python3
"""
Assumption:
.env file with username and password for API
.csv file with list of observations(images) to retrieve
Usage:
    python patrol_images.py --csv observations.csv
    python patrol_images.py --csv observations.csv --output ./my_images

Output filenames:
    {output}/{obs_id}_{camera}_{timestamp}.jpg
"""

import argparse
import csv
import os
import sys
import time
from pathlib import Path
from urllib.parse import urlparse
import json

import requests
from dotenv import load_dotenv
from requests.auth import HTTPBasicAuth

load_dotenv()

BASE_URL = "https://ascen.to/api/v2"
RETRY_ATTEMPTS = 2
RETRY_DELAY = 2


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download patrol images from Ascento using your observations CSV.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--csv", "-c", required=True,
                        help="Path to the observations CSV file")
    parser.add_argument("--output", "-d", default="./patrol_images",
                        help="Output directory for downloaded images (default: ./patrol_images)")
    parser.add_argument("--skip-existing", action="store_true", default=True,
                        help="Skip images that already exist on disk (default: true)")
    parser.add_argument("--no-skip-existing", action="store_false", dest="skip_existing",
                        help="Re-download even if files already exist")
    return parser.parse_args()


def load_csv(csv_path: str) -> list[dict]:
    """Load and validate the observations CSV."""
    path = Path(csv_path)
    if not path.exists():
        print(f"ERROR: CSV file not found: {csv_path}")
        sys.exit(1)

    required = ["ID", "OrgKey", "Timestamp", "ExpectedImages"]
    rows = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        headers = list(reader.fieldnames or [])
        for col in required:
            if col not in headers:
                print(f"ERROR: Required column '{col}' not found in CSV.")
                print(f"Found: {', '.join(headers)}")
                sys.exit(1)
        for row in reader:
            if row["ID"].strip():
                rows.append(row)
    return rows


def parse_cameras(raw: str) -> list[str]:
    """Parse ExpectedImages: handles ["front" "thermal"] and ["right"] formats."""
    import re
    raw = raw.strip()
    matches = re.findall(r'"([^"]+)"', raw)
    if matches:
        return matches
    return [c.strip() for c in raw.split(",") if c.strip()]


def camera_slug(camera: str) -> str:
    """'front thermal' -> 'front_thermal'"""
    return camera.lower().replace(" ", "_").replace("/", "-")


def fetch_observation(session: requests.Session, org: str, obs_id: str) -> dict:
    """Fetch observation details from API. Exits on any failure."""
    url = f"{BASE_URL}/organization/{org}/observation/{obs_id}"
    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            resp = session.get(url, timeout=30)
            if resp.status_code == 200:
                return resp.json()
            elif resp.status_code == 401:
                print("ERROR: Authentication failed. Check your username and password.")
                sys.exit(1)
            elif resp.status_code == 404:
                print(f"  [ERROR] Observation not found: {obs_id}")
                sys.exit(1)
            else:
                err = ""
                try:
                    err = resp.json().get("error", "")
                except Exception:
                    pass
                print(f"  [ERROR] HTTP {resp.status_code} {err} for {obs_id} (attempt {attempt}/{RETRY_ATTEMPTS})")
        except requests.RequestException as e:
            print(f"  [ERROR] Request failed for {obs_id}: {e} (attempt {attempt}/{RETRY_ATTEMPTS})")

        if attempt < RETRY_ATTEMPTS:
            time.sleep(RETRY_DELAY)
        else:
            print(f"\nAborting: could not fetch observation {obs_id}.")
            sys.exit(1)


def download_image(session: requests.Session, url: str, dest: Path) -> None:
    """Download one image to disk. Exits on failure."""
    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            resp = requests.get(url, timeout=60, stream=True)
            if resp.status_code == 200:
                dest.parent.mkdir(parents=True, exist_ok=True)
                with open(dest, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=8192):
                        f.write(chunk)
                return
            else:
                print(f"    [ERROR] HTTP {resp.status_code} downloading {dest.name} (attempt {attempt}/{RETRY_ATTEMPTS})")
        except requests.RequestException as e:
            print(f"    [ERROR] Download error for {dest.name}: {e} (attempt {attempt}/{RETRY_ATTEMPTS})")

        if attempt < RETRY_ATTEMPTS:
            time.sleep(RETRY_DELAY)
        else:
            print(f"\nAborting: failed to download {dest.name}.")
            print(f"  URL: {url}")
            sys.exit(1)


def image_ext(url: str) -> str:
    ext = Path(urlparse(url).path).suffix.lower()
    return ext if ext in (".jpg", ".jpeg", ".png", ".webp", ".tiff") else ".jpg"


def main():
    args = parse_args()

    username = os.environ.get("ASCENTO_USERNAME")
    password = os.environ.get("ASCENTO_PASSWORD")
    if not username or not password:
        print("ERROR: ASCENTO_USERNAME and ASCENTO_PASSWORD must be set in your .env file.")
        sys.exit(1)
    print(f"  CSV file   : {args.csv}")
    print(f"  Output dir : {args.output}")

    print("[1/3] Loading observations from CSV...")
    rows = load_csv(args.csv)
    print(f"  Found {len(rows)} observations.\n")

    if not rows:
        print("No observations found. Exiting.")
        sys.exit(0)

    session = requests.Session()
    session.auth = HTTPBasicAuth(username, password)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[2/3] Downloading images...\n")
    total_images = 0
    skipped_obs = 0

    for i, row in enumerate(rows, 1):
        obs_id    = row["ID"].strip()
        org       = row["OrgKey"].strip()
        timestamp = row["Timestamp"].strip()
        cameras   = parse_cameras(row["ExpectedImages"])

        print(f"  [{i}/{len(rows)}] {obs_id}  ts={timestamp}  cameras: {', '.join(cameras)}")

        # Skip if all expected files already exist
        if args.skip_existing:
            expected_files = [
                output_dir / f"{obs_id}_{camera_slug(cam)}_{timestamp}.jpg"
                for cam in cameras
            ]
            if all(f.exists() for f in expected_files):
                print(f"    [SKIP] All {len(expected_files)} file(s) already exist.\n")
                skipped_obs += 1
                continue


        # Fetch observation from API to get signed image URLs
        obs = fetch_observation(session, org, obs_id)
        api_images = obs.get("images", [])

        # Build lookup: normalised camera name -> image dict
        def norm(s):
            return s.strip().lower().replace("_", " ")

        api_by_camera = {norm(img.get("camera", "")): img for img in api_images}

        deleted = sum(1 for img in api_images if img.get("fileDeleted", False))
        if deleted:
            print(f"    {deleted} image(s) have been deleted from storage and will be skipped.")

        for cam in cameras:
            slug = camera_slug(cam)
            api_img = api_by_camera.get(norm(cam))

            if api_img is None:
                print(f"    [ERROR] Camera '{cam}' not found in API response for {obs_id}.")
                print(f"      Available: {[img.get('camera') for img in api_images]}")
                sys.exit(1)

            if api_img.get("fileDeleted", False):
                print(f"    [SKIP] {cam} — file deleted from storage.")
                continue

            url = api_img.get("url", "")
            if not url:
                print(f"    [ERROR] No URL for camera '{cam}' on observation {obs_id}. Aborting.")
                sys.exit(1)

            ext = image_ext(url)
            safe_ts = timestamp.replace(" ", "_").replace(",", "").replace(":", "-")
            dest = output_dir / f"{obs_id}_{slug}_{safe_ts}{ext}"

            download_image(session, url, dest)
            size_kb = dest.stat().st_size // 1024
            print(f"    ✓ {dest.name} ({size_kb} KB)")
            total_images += 1

        print()

    print(f"[3/3] Summary")
    print(f"  Observations processed : {len(rows) - skipped_obs}")
    print(f"  Observations skipped   : {skipped_obs} (already downloaded)")
    print(f"  Images downloaded      : {total_images}")
    print(f"  Output directory       : {output_dir.resolve()}")


if __name__ == "__main__":
    main()