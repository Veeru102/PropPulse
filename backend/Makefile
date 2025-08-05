.PHONY: run clean

SHELL := /bin/bash

run:
	@echo "Running PropPulse backend with FAISS support..."
	./run_with_faiss.sh

clean:
	@echo "Cleaning up...
	@find . -type f -name "*.pyc" -delete
	@find . -type d -name "__pycache__" -exec rm -rf {} +
	@rm -f Miniconda3-latest-MacOSX-arm64.sh
	@echo "Cleanup complete." 