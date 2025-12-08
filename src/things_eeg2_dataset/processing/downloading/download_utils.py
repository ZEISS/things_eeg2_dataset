import logging
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import gdown

logger = logging.getLogger(__name__)

PAGE_NOT_FOUND = 404


def download_from_url(  # noqa: PLR0912
    url: str,
    dest_path: Path,
    description: str = "file",
    dry_run: bool = False,
    max_retries: int = 3,
) -> bool:
    """Download a file with progress tracking and retry logic.

    Args:
        url: URL to download from
        dest_path: Destination file path
        description: Description for progress messages

    Returns:
        True if download succeeded, False otherwise
    """
    if dry_run:
        logger.info(f"[DRY RUN] Would download {description} from {url}")
        return True

    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"Downloading {description} (attempt {attempt}/{max_retries})")
            logger.info(f"URL: {url}")

            # Create request with headers
            request = Request(url)  # noqa: S310
            request.add_header("User-Agent", "Mozilla/5.0")

            # Open connection
            with urlopen(request, timeout=1000) as response:  # noqa: S310
                # Get file size if available
                file_size = response.headers.get("Content-Length")
                if file_size:
                    file_size = int(file_size)
                    logger.info(f"File size: {file_size / (1024 * 1024):.1f} MB")

                # Download with progress tracking
                dest_path.parent.mkdir(parents=True, exist_ok=True)

                downloaded = 0
                chunk_size = 8192
                last_progress = 0

                with dest_path.open("wb") as f:
                    while True:
                        chunk = response.read(chunk_size)
                        if not chunk:
                            break

                        f.write(chunk)
                        downloaded += len(chunk)

                        # Log progress every 10%
                        if file_size:
                            progress = int((downloaded / file_size) * 100)
                            if progress >= last_progress + 10:
                                logger.info(
                                    f"Progress: {progress}% ({downloaded / (1024 * 1024):.1f} MB)"
                                )
                                last_progress = progress

            logger.info(f"Successfully downloaded {description}")
            return True

        except HTTPError as e:
            logger.error(f"HTTP error downloading {description}: {e.code} {e.reason}")
            if e.code == PAGE_NOT_FOUND:
                logger.error(f"File not found at {url}")
                return False

        except URLError as e:
            logger.error(f"Network error downloading {description}: {e.reason}")

        except Exception as e:
            logger.error(f"Unexpected error downloading {description}: {e}")

        # Retry with exponential backoff
        if attempt < max_retries:
            wait_time = 2**attempt
            logger.info(f"Retrying in {wait_time} seconds...")
            time.sleep(wait_time)
        else:
            logger.error(
                f"Failed to download {description} after {max_retries} attempts"
            )
            return False

    return False


def download_from_gdrive(
    file_url: str,
    dest_path: Path,
    is_folder: bool = False,
) -> None:
    """Download a file from Google Drive.

    Args:
        file_id: Google Drive file ID
        dest_path: Destination file path
        is_folder: Whether the URL points to a folder
    Returns:
        True if download succeeded, False otherwise
    """
    try:
        # Ensure parent directory exists
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        if not is_folder:
            result = gdown.download(
                url=file_url, output=str(dest_path), quiet=False, fuzzy=True
            )
        else:
            result = gdown.download_folder(
                url=file_url, output=str(dest_path), quiet=False, use_cookies=False
            )

        # Check if download was successful
        if result is None:
            logger.error(f"Failed to download from Google Drive: {file_url}")
            raise RuntimeError("gdown reported failure")

        # Verify file exists and has reasonable size (not an HTML error page)
        if not dest_path.exists():
            logger.error(f"Downloaded file does not exist: {dest_path}")
            raise RuntimeError("Downloaded file missing")

        file_size = dest_path.stat().st_size
        # If file is less than 1MB, it might be an error page
        if file_size < 1024 * 1024:
            logger.warning(
                f"Downloaded file suspiciously small ({file_size} bytes), checking content"
            )
            # Check if it's an HTML file
            with dest_path.open("rb") as f:
                header = f.read(1024)
                if b"<!DOCTYPE html>" in header or b"<html" in header:
                    logger.error(
                        "Downloaded HTML instead of file. URL may be invalid or file not publicly accessible."
                    )
                    dest_path.unlink()  # Remove the bad file
                    raise RuntimeError("Downloaded HTML error page")

        logger.info(
            f"Successfully downloaded {dest_path.name} ({file_size / (1024 * 1024):.1f} MB)"
        )

    except Exception as e:
        logger.error(f"Error downloading {file_url} from Google Drive: {e}")
        if dest_path.exists():
            dest_path.unlink()  # Clean up partial download
        raise RuntimeError(f"Download failed: {e}") from e
