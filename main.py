import sys
import time
import webbrowser
import uvicorn
from api.app import app

def main():
    host = "127.0.0.1"
    port = 8000
    url = f"http://{host}:{port}"

    print("=" * 68)
    print("  🏥  PrivateMed: Differentially Private Clinical Model Training")
    print("=" * 68)
    print(f"  [+] Status      : Engine Initialized & RUNNING")
    print(f"  [+] Dashboard   : {url}")
    print(f"  [+] Framework   : FastAPI · PyTorch · Opacus DP-SGD · Chart.js")
    print(f"  [+] Accounting  : Rényi Differential Privacy (RDP) & Laplace")
    print("=" * 68)
    print(f"\n  👉  Open your browser and navigate to: {url}")
    print("  👉  Press CTRL + C in this terminal to shut down the server.\n")

    # Launch browser automatically
    try:
        webbrowser.open(url)
    except Exception:
        pass

    # Run Uvicorn server
    uvicorn.run(
        "api.app:app",
        host=host,
        port=port,
        reload=False,
        log_level="info"
    )

if __name__ == "__main__":
    main()
