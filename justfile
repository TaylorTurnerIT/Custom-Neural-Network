# Start the Marimo editor with MCP enabled
run:
    uv run --with "marimo[mcp]" marimo edit mlhw7_marimo.py --host localhost --port 2718 --mcp --no-token --watch