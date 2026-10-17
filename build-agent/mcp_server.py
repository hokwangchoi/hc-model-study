"""
The same read-only tools, exposed as an MCP server, so any MCP client (Claude Code, Claude Desktop,
the Claude Agent SDK) can use your notes without the hand-written loop in agent.py.

    pip install "mcp>=2"
    claude mcp add --transport stdio notes -- python /abs/path/to/build-agent/mcp_server.py

Write tools are left out on purpose (least privilege); the client decides which calls need approval.
"""

from mcp.server import MCPServer

from agent import HERE, Workspace

ws = Workspace(HERE / "sample_notes", HERE / "agent_data")
mcp = MCPServer("notes")


@mcp.tool()
def search_notes(query: str, k: int = 3) -> str:
    """Keyword (BM25) search over the owner's notes. Returns the best matching sections with file names."""
    return ws.search_notes(query, k)


@mcp.tool()
def read_note(path: str) -> str:
    """Read one note file, path relative to the notes folder."""
    return ws.read_note(path)


@mcp.tool()
def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression such as '0.175 * 2340'."""
    return ws.calculator(expression)


@mcp.tool()
def list_todos() -> str:
    """Show the owner's todo list."""
    return ws.list_todos()


if __name__ == "__main__":
    mcp.run()  # stdio transport: the client starts this process and talks over stdin/stdout
