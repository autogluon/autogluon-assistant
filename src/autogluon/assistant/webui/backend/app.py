# src/autogluon/assistant/webui/backend/app.py

import argparse

from flask import Flask

from .queue import get_queue_manager
from .routes import bp


def create_app(host: str = "0.0.0.0", port: int = 5000, debug: bool = False) -> Flask:
    """Create Flask app with configurable parameters."""
    app = Flask(__name__)

    # Store configuration as app attributes for access in other parts
    app.config["HOST"] = host
    app.config["PORT"] = port
    app.config["DEBUG"] = debug

    app.register_blueprint(bp, url_prefix="/api")

    # Initialize and start queue manager
    with app.app_context():
        queue_manager = get_queue_manager()
        queue_manager.start()

    return app


def main() -> None:
    """Console entry-point: `mlzero-backend`."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="MLZero Backend Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=5000, help="Port to bind to (default: 5000)")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--no-debug", dest="debug", action="store_false", help="Disable debug mode (default)")
    parser.set_defaults(debug=False)

    args = parser.parse_args()

    # Create and run app with parsed arguments
    app = create_app(host=args.host, port=args.port, debug=args.debug)
    app.run(host=args.host, port=args.port, debug=args.debug, use_reloader=False)


if __name__ == "__main__":
    main()
