"""Entry point for ``python -m dfdetect``. Forwards to the Typer CLI."""
from .cli import app

if __name__ == "__main__":
    app()
