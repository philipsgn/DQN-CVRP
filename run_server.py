"""Robust launcher for MP-MOTO-CVRP FastAPI Server.
Restricts WatchFiles reload monitoring strictly to src/web/ to prevent
unwanted reload storms when running tests, training, or evaluations.
"""
import sys
import uvicorn

if __name__ == "__main__":
    try:
        uvicorn.run(
            "src.web.server:app",
            host="127.0.0.1",
            port=8000,
            reload=True,
            reload_dirs=["src/web"],
            reload_excludes=[
                "*.log", "*.pt", "*.npy", "*.graphml", "*.json",
                "*.tmp", "*.pyc", "__pycache__", ".pytest_cache",
                "data/*", "models/*", "tests/*", "scripts/*", "docs/*"
            ],
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n[INFO] Server gracefully stopped by user.")
        sys.exit(0)

