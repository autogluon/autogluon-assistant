# ── src/autogluon/assistant/webui/backend/app.py ──────────────────────────────
from flask import Flask

from .config import Config
from .routes import bp


def create_app() -> Flask:
    app = Flask(__name__)
    app.config.from_object(Config)
    app.register_blueprint(bp, url_prefix="/api")
    return app


def main() -> None:
    """Console entry‑point:  `mlzero-backend`."""
    app = create_app()
    app.run(host=Config.HOST, port=Config.PORT, debug=Config.DEBUG, use_reloader=False)


if __name__ == "__main__":
    main()
