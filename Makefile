.PHONY: clean uv-sync

uv-clean:
	rm -rf build dist

uv-sync:
	uv sync --no-cache --no-build-isolation --extra build

