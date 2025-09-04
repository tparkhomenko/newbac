## Restarting Backend (FastAPI/Uvicorn) and Frontend (Vite)

Use these steps to cleanly stop any running servers and restart both the backend and the frontend during development.

### Prerequisites
- Python virtual environment created and dependencies installed
  - `python -m venv .venv`
  - `source .venv/bin/activate && pip install -r backend/requirements.txt`
- Frontend dependencies installed
  - `cd frontend && npm install`

### Quick Restart (copy/paste)
Run these in the project root: `/home/parkhomenko/Documents/new_project`

```bash
# 1) Stop any running servers (ignore errors if none running)
pkill -f 'uvicorn.*8000' || true
pkill -f 'backend.main:app' || true
pkill -f 'vite' || true
pkill -f 'node.*vite' || true
pkill -f 'npm.*dev' || true

# 2) Start backend (Uvicorn) in one terminal
source .venv/bin/activate
HOST=127.0.0.1 PORT=8000 uvicorn backend.main:app --host 127.0.0.1 --port 8000 --log-level info

# 3) Start frontend (Vite) in a second terminal
cd frontend
npm run dev
```

### Verification
- Backend health: `curl -s http://127.0.0.1:8000/healthz`
  - Expected: JSON with `status: ok`, `models` status, and `metadata_loaded: true`.
- Frontend: open `http://127.0.0.1:3000` (or `http://localhost:3000`) in a browser.

### Alternative: Background start (no logs)
```bash
# Backend (background)
source .venv/bin/activate && HOST=127.0.0.1 PORT=8000 \
uvicorn backend.main:app --host 127.0.0.1 --port 8000 --log-level info &

# Frontend (background)
cd frontend && npm run dev &
```

Stop background jobs with the same `pkill` commands from the Quick Restart section.

### Common Issues
- Port in use (8000 or 3000):
  - Check: `lsof -i :8000` or `lsof -i :3000`
  - Kill: `pkill -f 'uvicorn.*8000'` or `pkill -f 'vite'`
  - Change backend port: `uvicorn ... --port 8001` and access `http://127.0.0.1:8001/healthz`
- Missing dependency (e.g., `python-multipart`):
  - `source .venv/bin/activate && pip install -r backend/requirements.txt`
- CUDA issues:
  - Force CPU: `TORCH_DEVICE=cpu uvicorn backend.main:app --host 127.0.0.1 --port 8000`
- CORS/Network:
  - Ensure frontend calls `http://127.0.0.1:8000` (or matching host/port) and backend CORS is enabled (already configured).

### Useful Endpoints
- Health: `GET /healthz`
- Predict: `POST /predict` (`multipart/form-data` with `file`)
- Model info: `GET /model`
- Switch model: `POST /model/switch` with `{ "architecture": "parallel" | "multi" }`
- Quick test: `POST /quicktest`

### Notes
- Always run commands from the project root: `/home/parkhomenko/Documents/new_project`.
- Keep one terminal dedicated to the backend to watch logs during development.




