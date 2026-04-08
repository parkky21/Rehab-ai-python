# Rehab AI Python Server

This repository now contains two runtime modes:

- Desktop prototype: `app.py` (existing CustomTkinter flow)
- Production backend API: `api_server.main:app` (FastAPI + MongoDB + WebSocket)

## Run Full Stack Locally

1. Create `.env` in `python-server` from `.env.example`.
2. Create `.env` in `web-client` from `web-client/.env.example`.
3. Start MongoDB, API, and web client from the `python-server` folder:

```bash
docker compose up --build
```

3. Open API docs:

```text
http://localhost:8000/docs
```

4. Open web app:

```text
http://localhost:5173
```

## API Overview

- `POST /api/v1/auth/register` Doctor or Patient registration
- `POST /api/v1/auth/login` Login and token issuance
- `POST /api/v1/auth/refresh` Refresh access token
- `GET /api/v1/auth/me` Current profile
- `GET /api/v1/exercises` Exercise catalog from current scoring configs
- `GET /api/v1/doctor/patients` Doctor linked patient list
- `POST /api/v1/doctor/patients/link` Link patient to doctor
- `POST /api/v1/doctor/assignments` Assign exercise to linked patient
- `GET /api/v1/doctor/patients/{patient_id}/report` Doctor progress report
- `GET /api/v1/patient/assignments` Patient assignment queue
- `GET /api/v1/patient/sessions` Patient session history
- `GET /api/v1/patient/progress` Patient trend and adherence summary
- `WS /api/v1/ws/session?token=<JWT>&assignment_id=<id>` Real-time landmark stream and feedback

## Seed Demo Users

With API and Mongo running, create demo users and baseline assignments:

```bash
uv run python -m api_server.seed_demo
```

## WebSocket Frame Message

Send this payload at ~24-30 FPS:

```json
{
	"type": "landmark_frame",
	"frame_index": 1,
	"timestamp_ms": 1710000000000,
	"landmarks": [
		{ "x": 0.1, "y": 0.2, "z": 0.0, "visibility": 0.99 }
	]
}
```

Notes:

- You must send 33 landmarks (MediaPipe Pose output).
- Session can be ended by sending `{ "type": "session_end" }`.
