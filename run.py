"""
Launch the Leaf Detection app on localhost.

  python run.py

Then open http://127.0.0.1:8000 in your browser. Everything runs locally;
no internet is used at runtime.
"""
from __future__ import annotations

import webbrowser

import uvicorn

from app.backend import config


def main() -> None:
    url = f"http://{config.HOST}:{config.PORT}"
    print(f"Starting Leaf Detection at {url}")
    try:
        webbrowser.open(url)
    except Exception:
        pass
    uvicorn.run("app.backend.main:app", host=config.HOST, port=config.PORT,
                reload=False)


if __name__ == "__main__":
    main()
