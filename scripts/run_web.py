"""Launch the Solar Output web UI: http://127.0.0.1:8000"""
import sys
from pathlib import Path

import uvicorn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if __name__ == "__main__":
    uvicorn.run("web.server:app", host="127.0.0.1", port=8000, reload=False)
