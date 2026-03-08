import logging
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import gdown
from tqdm import tqdm

logger = logging.getLogger(__name__)


class NoURLFilter(logging.Filter):
    """Logging filter to exclude URL messages from logs."""

    def filter(self, record):  # noqa: ANN001, ANN201
        return (
            "URL:" not in record.getMessage()
            and "Following redirect to:" not in record.getMessage()
        )


logger.addFilter(NoURLFilter())

PAGE_NOT_FOUND = 404


def download_from_url(  # noqa: PLR0911, PLR0912, PLR0915
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

            # Create request with headers (enhanced for Figshare)
            request = Request(url)  # noqa: S310
            request.add_header(
                "User-Agent",
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            )
            request.add_header(
                "Accept",
                "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            )
            request.add_header("Accept-Language", "en-US,en;q=0.5")
            request.add_header(
                "Accept-Encoding", "identity"
            )  # Disable compression to get accurate file sizes
            request.add_header("Connection", "keep-alive")
            request.add_header("Upgrade-Insecure-Requests", "1")

            # Special handling for Figshare URLs
            if "figshare.com" in url:
                request.add_header("Referer", "https://figshare.com/")

            # Open connection with longer timeout for large files
            with urlopen(request, timeout=300) as response:  # noqa: S310
                # Handle redirects for Figshare
                actual_url = response.geturl()  # noqa: F841

                # Get file size if available
                file_size = response.headers.get("Content-Length")
                if file_size:
                    file_size = int(file_size)
                    logger.info(f"File size: {file_size / (1024 * 1024):.1f} MB")
                else:
                    logger.info("File size unknown")

                # Download with progress tracking
                dest_path.parent.mkdir(parents=True, exist_ok=True)

                chunk_size = 8192

                with dest_path.open("wb") as f:
                    with tqdm(
                        total=file_size,
                        unit="B",
                        unit_scale=True,
                        unit_divisor=1024,
                        desc=description,
                        ncols=80,
                        disable=file_size is None,
                    ) as pbar:
                        downloaded = 0
                        while True:
                            chunk = response.read(chunk_size)
                            if not chunk:
                                break
                            f.write(chunk)
                            downloaded += len(chunk)
                            pbar.update(len(chunk))

                    # If the file size was unknown, log total download
                    if file_size is None:
                        logger.info(f"Downloaded {downloaded / (1024 * 1024):.1f} MB")

            # Verify the downloaded file
            if dest_path.exists():
                final_size = dest_path.stat().st_size
                logger.info(
                    f"Successfully downloaded {description} ({final_size / (1024 * 1024):.1f} MB)"
                )

                # Check if file seems valid (not an error page)
                if (
                    final_size < 1024  # noqa: PLR2004
                ):  # Less than 1KB is suspicious for these data files
                    logger.warning(
                        f"Downloaded file is very small ({final_size} bytes), checking content"
                    )
                    with dest_path.open("rb") as f:
                        header = f.read(512)
                        if b"<!DOCTYPE html>" in header or b"<html" in header:
                            logger.error(
                                "Downloaded HTML error page instead of data file"
                            )
                            dest_path.unlink()  # Remove the bad file
                            return False

                return True
            else:
                logger.error("Downloaded file does not exist")
                return False

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
        file_url: Google Drive file URL
        dest_path: Destination file path
        is_folder: Whether the URL points to a folder

    Raises:
        RuntimeError: If the download fails
    """
    try:
        # Ensure parent directory exists
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        if not is_folder:
            from functools import partial  # noqa: PLC0415

            tqdm_partial = partial(
                tqdm, ncols=80, unit="B", unit_scale=True, unit_divisor=1024
            )

            result = gdown.download(
                url=file_url,
                output=str(dest_path),
                quiet=False,
                fuzzy=True,
                resume=True,
            )
        else:
            from functools import partial  # noqa: PLC0415

            tqdm_partial = partial(  # noqa: F841
                tqdm, ncols=80, unit="B", unit_scale=True, unit_divisor=1024
            )
            result = gdown.download_folder(
                url=file_url, output=str(dest_path), quiet=False, use_cookies=False
            )

        # Check if download was successful
        if result is None:
            raise RuntimeError("gdown reported failure")

        # Verify file exists and has reasonable size (not an HTML error page)
        if not dest_path.exists():
            raise RuntimeError("Downloaded file missing")

        file_size = dest_path.stat().st_size
        # If file is less than 1MB, it might be an error page
        if file_size < 1024 * 1024:
            with dest_path.open("rb") as f:
                header = f.read(512)
                if b"<!DOCTYPE html>" in header or b"<html" in header:
                    dest_path.unlink()  # Remove the bad file
                    raise RuntimeError(
                        "Downloaded HTML error page instead of data file"
                    )

        logger.info(
            f"Successfully downloaded {dest_path.name} ({file_size / (1024 * 1024):.1f} MB)"
        )

    except Exception as e:
        # Log the file_url only in case of failure
        logger.error(f"Error downloading from Google Drive: {e}")
        logger.error(f"Manual download may be required. URL: {file_url}")
        if dest_path.exists():
            dest_path.unlink()  # Clean up partial download
        raise RuntimeError(f"Download failed: {e}") from e
