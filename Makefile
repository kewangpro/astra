.PHONY: run stop ports

VENV          := .venv/bin
PORT          := 8200
PORT_FRONTEND := 3200
LOG_DIR       := logs

run:
	@mkdir -p $(LOG_DIR)
	@$(VENV)/uvicorn backend.main:app --host 0.0.0.0 --port $(PORT) --reload \
		> $(LOG_DIR)/backend.log 2>&1 & echo $$! > $(LOG_DIR)/backend.pid
	@cd frontend && npm run dev > ../$(LOG_DIR)/frontend.log 2>&1 & echo $$! > $(LOG_DIR)/frontend.pid
	@echo "Backend   :$(PORT)          →  tail -f $(LOG_DIR)/backend.log"
	@echo "Frontend  :$(PORT_FRONTEND)          →  tail -f $(LOG_DIR)/frontend.log"

stop:
	@lsof -ti :$(PORT) | xargs kill -9 2>/dev/null && echo "Stopped backend on :$(PORT)" || echo "Nothing on :$(PORT)"
	@lsof -ti :$(PORT_FRONTEND) | xargs kill -9 2>/dev/null && echo "Stopped frontend on :$(PORT_FRONTEND)" || echo "Nothing on :$(PORT_FRONTEND)"
	@rm -f $(LOG_DIR)/backend.pid $(LOG_DIR)/frontend.pid

ports:
	@echo "astra Port Status:"
	@echo "------------------------------------------------"
	@printf "%-22s %-6s %-10s %s\n" "SERVICE" "PORT" "STATUS" "PROCESS/PID"
	@echo "------------------------------------------------"
	@_check_port() { \
		svc=$$1; port=$$2; \
		pid=$$(lsof -ti :$$port -sTCP:LISTEN 2>/dev/null | head -1); \
		if [ -n "$$pid" ]; then \
			proc=$$(ps -p $$pid -o comm= 2>/dev/null | sed 's|.*/||'); \
			printf "%-22s %-6s %-10s %s\n" "$$svc" "$$port" "ACTIVE" "$$proc ($$pid)"; \
		else \
			printf "%-22s %-6s %-10s %s\n" "$$svc" "$$port" "free" "-"; \
		fi; \
	}; \
	_check_port "FastAPI Backend" $(PORT); \
	_check_port "Next.js Dashboard" $(PORT_FRONTEND)
	@echo "------------------------------------------------"
