"""Allow running as: python -m third_party.overlay_viz"""

import uvicorn

from .app import create_app


def main() -> None:
    app = create_app()
    uvicorn.run(app, host="0.0.0.0", port=8888)


if __name__ == "__main__":
    main()
