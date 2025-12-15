from __future__ import annotations

import datetime
import http.server
import os
import socket
import socketserver
import webbrowser
from collections.abc import Callable
from pathlib import Path

from rich import print  # noqa: A004
from rich.panel import Panel


def show_profiling(project_dir: Path, serve: bool, list_all: bool, port: int) -> None:
    profile_dir = project_dir / "profiling"

    if not profile_dir.exists():
        raise FileNotFoundError(f"No profiling directory found at {profile_dir}")

    # Get all html files sorted by modification time (newest first)
    reports = sorted(profile_dir.glob("*.html"), key=os.path.getmtime, reverse=True)

    if not reports:
        raise FileNotFoundError(f"No profiling reports found in {profile_dir}")

    if serve:
        os.chdir(profile_dir)  # Change cwd to serve files relative to here

        # Create a simple handler that lists directory
        Handler = http.server.SimpleHTTPRequestHandler
        local_ip = get_local_ip()
        localhost_url = f"http://localhost:{port}"
        network_url = f"http://{local_ip}:{port}"
        print(
            Panel(
                f"[bold]Serving Profiling Reports[/bold]\n"
                f"[dim]Directory: {profile_dir}[/dim]\n\n"
                f"🏠 [bold green]Local:[/bold green]   {localhost_url}\n"
                f"📡 [bold cyan]Network:[/bold cyan] {network_url}\n\n"
                f"[yellow]Press Ctrl+C to stop.[/yellow]",
                title="Web Server Running",
                border_style="green",
                expand=False,
            )
        )

        # Bind to "0.0.0.0" to allow external connections (important!)
        try:
            with socketserver.TCPServer(("0.0.0.0", port), Handler) as httpd:  # noqa: S104
                httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n[yellow]Server stopped.[/yellow]")
        return

    if list_all:
        print(f"[bold]Found {len(reports)} reports in {profile_dir}:[/bold]")
        for r in reports:
            print(
                f" - {r.name} ({datetime.datetime.fromtimestamp(r.stat().st_mtime, datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S')})"
            )
        return

    # Open the newest one
    latest_report = reports[0]
    print(f"[green]Opening latest report:[/green] {latest_report.name}")
    webbrowser.open(latest_report.as_uri())


def get_local_ip() -> str:
    """
    Best-effort attempt to get the local network IP address.
    Uses a dummy UDP connection to determine the primary interface.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # We don't actually send data, so 10.255.255.255 (unroutable) works fine.
        # We just want the OS to tell us which interface it would use.
        s.connect(("10.255.255.255", 1))
        ip = s.getsockname()[0]
    except Exception:
        # Fallback if no network is available
        ip = "127.0.0.1"
    finally:
        s.close()
    return ip


def run_with_profiling(
    fn: Callable[[], None],
    *,
    output_dir: Path,
    label: str,
    open_report: bool = True,
) -> None:
    """
    Run a callable under pyinstrument profiling and save an HTML report.

    Parameters
    ----------
    fn
        Zero-argument callable to execute.
    output_dir
        Directory where profiling reports are stored.
    label
        Label used in the output filename (e.g. "preprocessing").
    open_report
        Whether to auto-open the report in a browser (if not headless).
    """
    from pyinstrument import Profiler  # noqa: PLC0415

    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.datetime.now(tz=datetime.timezone.utc).strftime(
        "%Y%m%d_%H%M%S"
    )

    output_path = output_dir / f"{timestamp}_{label}.html"

    with Profiler() as profiler:
        fn()

    output_path.write_text(profiler.output_html())

    print(f"\n[green]Profiling report saved to:[/green] {output_path}")

    _maybe_open_report(output_path, open_report=open_report)


def _maybe_open_report(path: Path, *, open_report: bool) -> None:
    if not open_report:
        return

    is_headless = os.environ.get("DISPLAY") is None

    if is_headless:
        print("[yellow]Headless environment detected. Skipping auto-open.[/yellow]")
        print(
            "Run [bold cyan]things-eeg2 view-profile --serve[/bold cyan] to view it remotely."
        )
        return

    print("[yellow]Opening report in browser...[/yellow]")
    webbrowser.open(path.as_uri())
