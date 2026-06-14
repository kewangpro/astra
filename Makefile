.PHONY: run stop run-frontend stop-frontend ports

VENV         := .venv/bin
PORT         := 8200
PORT_FRONTEND := 3200

run:
	$(VENV)/uvicorn backend.main:app --host 0.0.0.0 --port $(PORT) --reload

stop:
	@lsof -ti :$(PORT) | xargs kill -9 2>/dev/null && echo "Stopped process on :$(PORT)" || echo "Nothing running on :$(PORT)"

run-frontend:
	cd frontend && npm run dev

stop-frontend:
	@lsof -ti :$(PORT_FRONTEND) | xargs kill -9 2>/dev/null && echo "Stopped process on :$(PORT_FRONTEND)" || echo "Nothing running on :$(PORT_FRONTEND)"

ports:
	@echo "astra Port Status:"
	@echo "------------------------------------------------"
	@printf "%-22s %-6s %-10s %s\n" "SERVICE" "PORT" "STATUS" "PROCESS/PID"
	@echo "------------------------------------------------"
	@_check_port() { \
		svc=$$1; port=$$2; \
		pid=$$(lsof -ti :$$port 2>/dev/null | head -1); \
		if [ -n "$$pid" ]; then \
			proc=$$(ps -p $$pid -o comm= 2>/dev/null); \
			printf "%-22s %-6s %-10s %s\n" "$$svc" "$$port" "ACTIVE" "$$proc (PID: $$pid)"; \
		else \
			printf "%-22s %-6s %-10s %s\n" "$$svc" "$$port" "free" "-"; \
		fi; \
	}; \
	_check_port "FastAPI Backend" $(PORT); \
	_check_port "Next.js Dashboard" $(PORT_FRONTEND)
	@echo "------------------------------------------------"
