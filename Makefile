.PHONY: up down prod logs migrate revision

# Start the dev stack (Vite + hot reload).
up:
	docker compose up --build

# Stop and remove containers.
down:
	docker compose down

# Production-style stack (nginx serves compiled SPA).
prod:
	docker compose -f docker-compose.yml -f docker-compose.prod.yml up --build

logs:
	docker compose logs -f

# Apply migrations manually (the migrate init container does this on startup).
migrate:
	docker compose run --rm migrate alembic upgrade head

# Autogenerate a new migration from model changes:
#   make revision rev=0002 m="add users table"   ->  0002_add_users_table.py
# The rev id is the filename prefix; pad it (0002, 0003, ...) to keep them ordered.
revision:
	docker compose run --rm migrate alembic revision --autogenerate --rev-id "$(rev)" -m "$(m)"
