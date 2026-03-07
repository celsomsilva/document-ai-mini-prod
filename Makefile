# Mark these targets as phony (not real files)
.PHONY: train run test docker docker-build docker-up docker-clean clean

train:
	python3 -m mlops_api.train

run:
	python3 -m uvicorn mlops_api.api:app --reload

test:
	python3 -m pytest -v

# Clean local builds, fast CI builds
docker-build:
ifeq ($(CI),true)
	docker compose build
else
	docker compose build --no-cache
endif

docker-up:
	docker compose up

docker: docker-build docker-up

# Docker cleanup (safe)
docker-clean:
	docker compose down --remove-orphans
	docker system prune -f


# Local cleanup (Python / ML caches)
# WARNING: aggressive cleanup (will slow next installs)
clean:
	rm -rf __pycache__ .pytest_cache
	rm -rf ~/.cache/pip
	rm -rf ~/.cache/torch
	rm -rf ~/.cache/huggingface

