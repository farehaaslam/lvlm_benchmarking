from pathlib import Path
import gdown


# ============================================================
# CONFIG
# ============================================================

# Google Drive folder URL
DRIVE_FOLDER_URL = "https://drive.google.com/drive/folders/1q-FLguwlKZezS5hjbdG9n0nXTY8sQPX1"

# Local destination
OUTPUT_DIR = Path("dataset/videos")


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# DOWNLOAD
# ============================================================

print("=" * 70)
print("Downloading videos from Google Drive")
print("=" * 70)

gdown.download_folder(
    DRIVE_FOLDER_URL,
    output=str(OUTPUT_DIR),
    quiet=False,
    use_cookies=False,
)


print("\n" + "=" * 70)
print("Download completed")
print("=" * 70)


# ============================================================
# SHOW DOWNLOADED FILES
# ============================================================

videos = sorted(OUTPUT_DIR.glob("*.mp4"))

print(f"\nVideos found: {len(videos)}")

for video in videos:
    size_mb = video.stat().st_size / (1024 * 1024)

    print(
        f"{video.name:<30} "
        f"{size_mb:>10.2f} MB"
    )