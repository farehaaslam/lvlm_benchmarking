from pathlib import Path
import gdown


# ============================================================
# CONFIG
# ============================================================

# Google Drive folder URL
DRIVE_FOLDER_URL_1 = "https://drive.google.com/drive/folders/1NDmn6g81VVMuPpGvOVXnMZk6ensAcRv5?usp=sharing"
DRIVE_FOLDER_URL_2="https://drive.google.com/drive/folders/1xscRQAYgSBrTZxTEOojyDtoD5_lDWuRc?usp=sharing"

# Local destination
OUTPUT_DIR = Path("dataset/questions")


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# DOWNLOAD
# ============================================================

print("=" * 70)
print("Downloading questions from Google Drive")
print("=" * 70)

gdown.download_folder(
    DRIVE_FOLDER_URL_1,
    output=str(OUTPUT_DIR),
    quiet=False,
    use_cookies=False,
)
print("A batch")
gdown.download_folder(
    DRIVE_FOLDER_URL_2,
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

questions = sorted(OUTPUT_DIR.glob("*.csv"))

print(f"\nQuestions found: {len(questions)}")

for question in questions:
    size_kb = question.stat().st_size / 1024

    print(
        f"{question.name:<30} "
        f"{size_kb:>10.2f} KB"
    )