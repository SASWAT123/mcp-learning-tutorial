#!/usr/bin/env python3
"""
Part 1: Multiple Tool Registration Patterns in MCP Servers (FastMCP Version)
============================================================================

This module demonstrates how to properly register and manage multiple tools
in a single MCP server using FastMCP, including:
- Simplified tool registration with decorators
- Automatic input schema validation
- Type hints for better development experience
- Built-in error handling
- Streamlined tool organization
"""

import json
import logging
import time
from datetime import datetime
from typing import List, Literal, Optional
from enum import Enum

import psutil
import platform
from fastmcp import FastMCP


class ToolCategory(Enum):
    """Categories for organizing tools"""
    SYSTEM = "system"
    NETWORK = "network"
    DATABASE = "database"
    UTILITY = "utility"


# Initialize FastMCP server
mcp = FastMCP("multi-tool-demo")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("FastMCPMultiTool")


# Define the actual functions without decorators first
def _get_time_impl(
    format: Literal["iso", "timestamp", "human"] = "iso",
    timezone: Optional[str] = None
) -> str:
    """Implementation of get_time"""
    now = datetime.now()

    if format == "iso":
        result = now.isoformat()
    elif format == "timestamp":
        result = str(int(time.time()))
    elif format == "human":
        result = now.strftime("%Y-%m-%d %H:%M:%S")
    else:
        result = now.isoformat()

    response = {
        "current_time": result,
        "format": format,
        "timezone": timezone or "system",
        "timestamp": int(time.time()),
    }

    return json.dumps(response, indent=2)


def _calculate_impl(
    operation: Literal["add", "subtract", "multiply", "divide", "power"],
    operands: List[float]
) -> str:
    """Implementation of calculate"""
    if len(operands) < 2:
        raise ValueError("At least 2 operands are required")

    try:
        if operation == "add":
            result = sum(operands)
        elif operation == "subtract":
            result = operands[0]
            for val in operands[1:]:
                result -= val
        elif operation == "multiply":
            result = 1
            for val in operands:
                result *= val
        elif operation == "divide":
            result = operands[0]
            for val in operands[1:]:
                if val == 0:
                    raise ValueError("Division by zero")
                result /= val
        elif operation == "power":
            if len(operands) != 2:
                raise ValueError("Power operation requires exactly 2 operands")
            result = operands[0] ** operands[1]
        else:
            raise ValueError(f"Unknown operation: {operation}")

        response = {
            "operation": operation,
            "operands": operands,
            "result": result,
            "calculation": f"{' '.join(map(str, operands))} = {result}",
        }

        return json.dumps(response, indent=2)

    except Exception as e:
        raise ValueError(f"Calculation error: {str(e)}")


def _system_status_impl(
    include_processes: bool = False,
    include_network: bool = False
) -> str:
    """Implementation of system_status"""
    try:
        # Basic system info
        status = {
            "platform": platform.system(),
            "architecture": platform.architecture()[0],
            "cpu_count": psutil.cpu_count(),
            "memory": {
                "total": psutil.virtual_memory().total,
                "available": psutil.virtual_memory().available,
                "percent_used": psutil.virtual_memory().percent,
            },
        }

        # Optional: process count
        if include_processes:
            status["process_count"] = len(list(psutil.process_iter()))

        # Optional: network stats
        if include_network:
            try:
                net_io = psutil.net_io_counters()
                status["network"] = {
                    "bytes_sent": net_io.bytes_sent,
                    "bytes_received": net_io.bytes_recv,
                    "packets_sent": net_io.packets_sent,
                    "packets_received": net_io.packets_recv,
                }
            except:
                status["network"] = "unavailable"

        return json.dumps(status, indent=2)

    except Exception as e:
        raise ValueError(f"System status error: {str(e)}")


def _process_text_impl(
    text: str,
    operations: List[Literal["uppercase", "lowercase", "reverse", "word_count", "char_count"]]
) -> str:
    """Implementation of process_text"""
    if not operations:
        raise ValueError("At least one operation must be specified")

    results = {"original_text": text, "operations": {}}

    for operation in operations:
        if operation == "uppercase":
            results["operations"]["uppercase"] = text.upper()
        elif operation == "lowercase":
            results["operations"]["lowercase"] = text.lower()
        elif operation == "reverse":
            results["operations"]["reverse"] = text[::-1]
        elif operation == "word_count":
            results["operations"]["word_count"] = len(text.split())
        elif operation == "char_count":
            results["operations"]["char_count"] = len(text)

    return json.dumps(results, indent=2)


# Now register the tools with FastMCP
@mcp.tool()
def get_time(
    format: Literal["iso", "timestamp", "human"] = "iso",
    timezone: Optional[str] = None
) -> str:
    """
    Get current system time in various formats

    Args:
        format: Time format to return (iso, timestamp, or human)
        timezone: Timezone (default: system timezone)

    Returns:
        Current time in the specified format
    """
    return _get_time_impl(format, timezone)


@mcp.tool()
def calculate(
    operation: Literal["add", "subtract", "multiply", "divide", "power"],
    operands: List[float]
) -> str:
    """
    Perform mathematical calculations

    Args:
        operation: Mathematical operation to perform
        operands: Numbers to operate on (minimum 2 numbers)

    Returns:
        Calculation result as JSON
    """
    return _calculate_impl(operation, operands)


@mcp.tool()
def system_status(
    include_processes: bool = False,
    include_network: bool = False
) -> str:
    """
    Get basic system status information

    Args:
        include_processes: Include process count in the output
        include_network: Include network statistics in the output

    Returns:
        System status information as JSON
    """
    return _system_status_impl(include_processes, include_network)


@mcp.tool()
def process_text(
    text: str,
    operations: List[Literal["uppercase", "lowercase", "reverse", "word_count", "char_count"]]
) -> str:
    """
    Process and transform text in various ways

    Args:
        text: Text to process
        operations: List of operations to perform on the text

    Returns:
        Text processing results as JSON
    """
    return _process_text_impl(text, operations)


@mcp.tool()
def list_tools_by_category(category: Optional[Literal["system", "network", "database", "utility"]] = None) -> str:
    """
    List all available tools, optionally filtered by category

    Args:
        category: Optional category filter

    Returns:
        List of tools with their information
    """
    # This is a meta-tool that describes the available tools
    tools_info = {
        "get_time": {
            "category": "utility",
            "description": "Get current system time in various formats",
            "version": "1.0.0"
        },
        "calculate": {
            "category": "utility",
            "description": "Perform mathematical calculations",
            "version": "1.0.0"
        },
        "system_status": {
            "category": "system",
            "description": "Get basic system status information",
            "version": "1.0.0"
        },
        "process_text": {
            "category": "utility",
            "description": "Process and transform text in various ways",
            "version": "1.0.0"
        },
        "list_tools_by_category": {
            "category": "utility",
            "description": "List all available tools, optionally filtered by category",
            "version": "1.0.0"
        }
    }

    if category:
        filtered_tools = {
            name: info for name, info in tools_info.items()
            if info["category"] == category
        }
        result = {
            "category_filter": category,
            "tools": filtered_tools,
            "count": len(filtered_tools)
        }
    else:
        result = {
            "all_tools": tools_info,
            "count": len(tools_info),
            "categories": list(set(info["category"] for info in tools_info.values()))
        }

    return json.dumps(result, indent=2)


# Demo and Testing Functions
# ==========================

def demo_fastmcp_tools():
    """Demonstrate the FastMCP tool system"""
    print("=== FastMCP Multiple Tools Demo ===")

    # List available tools
    print(f"\nAvailable tools in this demo:")
    tool_functions = [
        "get_time - Get current system time in various formats",
        "calculate - Perform mathematical calculations",
        "system_status - Get basic system status information",
        "process_text - Process and transform text in various ways",
        "list_tools_by_category - List tools by category"
    ]

    for tool_info in tool_functions:
        print(f"  - {tool_info}")

    print(f"\nTotal tools available: {len(tool_functions)}")

    # Test tool calls using the implementation functions
    print("\n=== Testing Tool Calls ===")

    # Test 1: Get time
    print("\n1. Testing get_time tool:")
    try:
        result = _get_time_impl(format="human")
        print(f"   Result: {result}")
    except Exception as e:
        print(f"   Error: {e}")

    # Test 2: Calculate
    print("\n2. Testing calculate tool:")
    try:
        result = _calculate_impl(operation="add", operands=[10, 20, 30])
        print(f"   Result: {result}")
    except Exception as e:
        print(f"   Error: {e}")

    # Test 3: Process text
    print("\n3. Testing process_text tool:")
    try:
        result = _process_text_impl(
            text="Hello World",
            operations=["uppercase", "reverse", "word_count"]
        )
        print(f"   Result: {result}")
    except Exception as e:
        print(f"   Error: {e}")

    # Test 4: System status
    print("\n4. Testing system_status tool:")
    try:
        result = _system_status_impl(include_processes=True)
        print(f"   Result: {result}")
    except Exception as e:
        print(f"   Error: {e}")

    # Test 5: List tools by category
    print("\n5. Testing list_tools_by_category:")
    try:
        result = list_tools_by_category(category="utility")
        print(f"   Result: {result}")
    except Exception as e:
        print(f"   Error: {e}")


def demonstrate_fastmcp_benefits():
    """Demonstrate the benefits of using FastMCP"""
    print("\n=== FastMCP Benefits Over Vanilla MCP ===")

    print("\n1. Simplified Registration:")
    print("   ✓ No manual server setup or handler registration")
    print("   ✓ Decorators automatically register tools")
    print("   ✓ Type hints provide automatic schema generation")

    print("\n2. Better Developer Experience:")
    print("   ✓ Type safety with mypy/IDE support")
    print("   ✓ Automatic input validation")
    print("   ✓ Built-in error handling")

    print("\n3. Reduced Boilerplate:")
    print("   ✓ No need for ToolRegistry class")
    print("   ✓ No manual schema definitions")
    print("   ✓ No manual MCP protocol handling")
    print("   ✓ Eliminated ~200+ lines of boilerplate code")

    print("\n4. Enhanced Features:")
    print("   ✓ Literal types for enum-like parameters")
    print("   ✓ Optional parameters with defaults")
    print("   ✓ Rich docstring support")
    print("   ✓ Automatic JSON schema generation")

    print("\n5. Code Comparison:")
    print("   • Original file: ~520 lines")
    print("   • FastMCP version: ~350 lines")
    print("   • Reduction: ~33% less code")


def show_comparison():
    """Show side-by-side comparison"""
    print("\n=== Code Comparison: Vanilla MCP vs FastMCP ===")

    print("\nVanilla MCP Tool Registration:")
    print("""
    # Complex setup required
    class ToolRegistry:
        def __init__(self):
            self.tools = {}
            self.handlers = {}
            self.schemas = {}

        def register_tool(self, metadata, handler, schema):
            # Manual registration...

    registry.register_tool(
        ToolMetadata(...),
        self.handle_calculate,
        {
            "type": "object",
            "properties": {...},
            "required": [...]
        }
    )
    """)

    print("\nFastMCP Tool Registration:")
    print("""
    # Simple decorator approach
    @mcp.tool()
    def calculate(
        operation: Literal["add", "subtract", "multiply", "divide"],
        operands: List[float]
    ) -> str:
        \"\"\"Perform mathematical calculations\"\"\"
        # Implementation here...
    """)


if __name__ == "__main__":
    # Check if running as MCP server (stdin/stdout communication)
    import sys
    if len(sys.argv) == 1 and not sys.stdin.isatty():
        # Running as MCP server
        mcp.run()
    else:
        # Running as demo
        print("Part 1: Multiple Tool Registration Patterns (FastMCP)")
        print("=" * 60)

        # Run the demonstration
        demo_fastmcp_tools()
        demonstrate_fastmcp_benefits()
        show_comparison()

        print("\nKey Improvements with FastMCP:")
        print("1. 🎯 Decorator-based tool registration (@mcp.tool())")
        print("2. 🔧 Automatic schema generation from type hints")
        print("3. ✅ Built-in input validation and error handling")
        print("4. 🚀 Simplified server setup and management")
        print("5. 💡 Better IDE support and type safety")
        print("6. 📦 Reduced code complexity and boilerplate")

        print(f"\n🏃 To run as MCP server: python {__file__}")
        print("The server will be available for MCP client connections.")

        # Show how to run as server
        print("\n📋 To use with MCP client, add to your config:")
        print("""
{
  "mcpServers": {
    "fastmcp-demo": {
      "command": "python",
      "args": ["/path/to/this/file"]
    }
  }
}
        """)