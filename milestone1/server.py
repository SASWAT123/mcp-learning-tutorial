# server.py
from mcp.server.fastmcp import FastMCP

# Name your server anything you like
mcp = FastMCP("EchoHealthServer")


@mcp.tool()
def echo(message: str) -> str:
    """Echo back whatever the user sent."""
    return message


@mcp.tool()
def health() -> str:
    """Simple health check."""
    return "ok"


@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


if __name__ == "__main__":
    # Run over stdio so MCP clients (Claude Desktop, VS Code, Inspector) can connect
    mcp.run()
