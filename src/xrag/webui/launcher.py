"""CLI launcher for the XRAG WebUI.

Boots a uvicorn process that serves the FastAPI backend defined in
``server.py`` and the static frontend assets in ``static/``.
"""

from __future__ import annotations

import argparse
import os
import sys
import webbrowser


def run_web_ui(host: str = "127.0.0.1", port: int = 8765, open_browser: bool = True) -> None:
    """Start the XRAG WebUI server.

    Args:
        host: bind host. Defaults to loopback so the UI is not exposed
            publicly by default.
        port: bind port.
        open_browser: if True, the default browser is opened at the UI URL.
    """
    import warnings

    # Silence known-benign third-party import warnings so the startup log
    # stays readable. These are emitted by deepeval / llama_index / pydantic
    # during the heavy import chain and do not affect functionality. Only
    # specific messages are filtered -- real warnings remain visible.
    warnings.filterwarnings("ignore", message=r".*pkg_resources is deprecated.*")
    warnings.filterwarnings("ignore", message=r".*You are using deepeval version.*")
    warnings.filterwarnings("ignore", message=r".*lazily loaded.*")
    try:
        from pydantic import UnsupportedFieldAttributeWarning

        warnings.filterwarnings("ignore", category=UnsupportedFieldAttributeWarning)
    except Exception:  # pragma: no cover - depends on pydantic internals
        pass

    try:
        import uvicorn  # type: ignore
    except ImportError as exc:  # pragma: no cover - requirements guard
        sys.stderr.write(
            "uvicorn is required for the XRAG WebUI. Install it with "
            "`pip install uvicorn`.\n"
        )
        raise SystemExit(1) from exc

    # When the launcher runs from inside the package, ``server`` is a sibling
    # module. We import it lazily so that ``xrag-cli help`` stays cheap.
    from xrag.webui import server  # noqa: F401 - imported for side effects

    url = f"http://{host}:{port}/"
    sys.stdout.write(f"\n  XRAG WebUI is starting at {url}\n  Press Ctrl+C to stop.\n\n")
    sys.stdout.flush()
    if open_browser and os.environ.get("XRAG_WEBUI_NO_BROWSER") != "1":
        try:
            webbrowser.open(url)
        except Exception:
            pass

    uvicorn.run(server.app, host=host, port=port, log_level="info")


def _main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="XRAG WebUI launcher")
    parser.add_argument("--host", default="127.0.0.1", help="bind host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8765, help="bind port (default: 8765)")
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="do not open the default browser automatically",
    )
    args = parser.parse_args(argv)
    run_web_ui(host=args.host, port=args.port, open_browser=not args.no_browser)


if __name__ == "__main__":
    _main()