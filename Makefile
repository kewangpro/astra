.PHONY: run stop ports

# Color codes
RED    := \033[0;31m
GREEN  := \033[0;32m
YELLOW := \033[0;33m
CYAN   := \033[0;36m
BOLD   := \033[1m
RESET  := \033[0m

VENV          := .venv/bin
PORT          := 8200
PORT_FRONTEND := 3200
PORT_MLX      := 8080
LOG_DIR       := logs

run:
	@mkdir -p $(LOG_DIR)
	@printf "\n$(BOLD)$(CYAN)Starting ASTRA services...$(RESET)\n"
	@$(VENV)/uvicorn backend.main:app --host 0.0.0.0 --port $(PORT) --reload --reload-dir backend \
		> $(LOG_DIR)/backend.log 2>&1 & echo $$! > $(LOG_DIR)/backend.pid
	@printf "$(GREEN)→ Backend$(RESET)   (http://localhost:$(PORT))  tail -f $(LOG_DIR)/backend.log\n"
	@cd frontend && npm run dev > ../$(LOG_DIR)/frontend.log 2>&1 & echo $$! > $(LOG_DIR)/frontend.pid
	@printf "$(GREEN)→ Frontend$(RESET)  (http://localhost:$(PORT_FRONTEND))  tail -f $(LOG_DIR)/frontend.log\n"
	@printf "\n$(BOLD)Status:$(RESET) make ports\n\n"

stop:
	@printf "$(BOLD)$(YELLOW)Stopping ASTRA services...$(RESET)\n"
	@lsof -ti :$(PORT) | xargs kill -9 2>/dev/null && printf "$(RED)✓ Backend stopped$(RESET) (:$(PORT))\n" || printf "$(YELLOW)- Nothing on :$(PORT)$(RESET)\n"
	@lsof -ti :$(PORT_FRONTEND) | xargs kill -9 2>/dev/null && printf "$(RED)✓ Frontend stopped$(RESET) (:$(PORT_FRONTEND))\n" || printf "$(YELLOW)- Nothing on :$(PORT_FRONTEND)$(RESET)\n"
	@rm -f $(LOG_DIR)/backend.pid $(LOG_DIR)/frontend.pid

ports:
	@printf "\n$(BOLD)$(CYAN)🔍 ASTRA Port Status:$(RESET)\n"
	@printf "$(BOLD)%-22s %-6s %-10s %s$(RESET)\n" "SERVICE" "PORT" "STATUS" "PROCESS/PID"
	@printf '%0.s-' {1..58}; printf '\n'
	@_check_port() { \
		svc=$$1; port=$$2; \
		pid=$$(lsof -ti :$$port -sTCP:LISTEN 2>/dev/null | head -1); \
		if [ -n "$$pid" ]; then \
			proc=$$(ps -p $$pid -o comm= 2>/dev/null | sed 's|.*/||'); \
			printf "$(GREEN)%-22s %-6s %-10s %s$(RESET)\n" "$$svc" "$$port" "ACTIVE" "$$proc ($$pid)"; \
		else \
			printf "$(RED)%-22s %-6s %-10s %s$(RESET)\n" "$$svc" "$$port" "free" "-"; \
		fi; \
	}; \
	_check_port "FastAPI Backend" $(PORT); \
	_check_port "Next.js Dashboard" $(PORT_FRONTEND); \
	_check_port "MLX LM Server" $(PORT_MLX)
	@printf '%0.s-' {1..58}; printf '\n\n'
