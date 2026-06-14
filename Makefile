.PHONY: run stop ports

VENV := .venv/bin
PORT := 8000

run:
	$(VENV)/uvicorn backend.main:app --host 0.0.0.0 --port $(PORT) --reload

stop:
	@lsof -ti :$(PORT) | xargs kill -9 2>/dev/null && echo "Stopped process on :$(PORT)" || echo "Nothing running on :$(PORT)"

ports:
	@lsof -iTCP -sTCP:LISTEN -P | grep -E "LISTEN" | awk '{print $$1, $$2, $$9}' | column -t
