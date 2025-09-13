#!/usr/bin/env python3
"""
Part 2: Platform-Specific Implementations with FastMCP
======================================================

This module demonstrates how to create cross-platform MCP servers using FastMCP,
including:
- Platform detection and abstraction layers
- OS-specific implementations with fallbacks
- Resource management across different operating systems
- Performance optimization per platform
- Error handling for platform-specific features

Using modern toolchain:
- UV for package management
- FastMCP for streamlined MCP development
- Type hints for better code quality
"""

import asyncio
import json
import logging
import platform
import subprocess
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Union
import psutil

# FastMCP imports - much cleaner than traditional MCP
from fastmcp import FastMCP


class PlatformType(Enum):
    """Supported platform types"""

    LINUX = "linux"
    MACOS = "darwin"
    WINDOWS = "windows"
    UNKNOWN = "unknown"


@dataclass
class PlatformInfo:
    """Comprehensive platform information"""

    platform_type: PlatformType
    os_name: str
    os_version: str
    architecture: str
    kernel_version: str
    python_version: str
    has_admin_rights: bool
    package_manager: Optional[str] = None
    shell: Optional[str] = None
    desktop_environment: Optional[str] = None


@dataclass
class SystemMetrics:
    """Cross-platform system metrics"""

    cpu_usage: float
    memory_usage: Dict[str, Any]
    disk_usage: Dict[str, Any]
    network_stats: Dict[str, Any]
    process_count: int
    uptime_seconds: float
    load_average: Optional[List[float]] = None
    temperature: Optional[Dict[str, Any]] = None


class PlatformProvider(Protocol):
    """Protocol defining platform-specific operations"""

    async def get_platform_info(self) -> PlatformInfo:
        """Get platform-specific information"""
        ...

    async def get_system_metrics(self) -> SystemMetrics:
        """Get system metrics in a platform-appropriate way"""
        ...

    async def list_running_services(self) -> List[Dict[str, Any]]:
        """List system services (varies greatly by platform)"""
        ...

    async def get_network_interfaces(self) -> List[Dict[str, Any]]:
        """Get network interface information"""
        ...

    async def check_admin_privileges(self) -> bool:
        """Check if running with administrative privileges"""
        ...


class BasePlatformProvider(ABC):
    """Base class for platform providers with common functionality"""

    def __init__(self):
        self.logger = logging.getLogger(f"{self.__class__.__name__}")

    async def get_common_metrics(self) -> Dict[str, Any]:
        """Get metrics that work across all platforms"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)

            # Memory usage
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()

            # Disk usage (root/C: drive)
            disk_usage = psutil.disk_usage(
                "/" if platform.system() != "Windows" else "C:\\"
            )

            # Network I/O
            try:
                net_io = psutil.net_io_counters()
                network_stats = {
                    "bytes_sent": net_io.bytes_sent,
                    "bytes_recv": net_io.bytes_recv,
                    "packets_sent": net_io.packets_sent,
                    "packets_recv": net_io.packets_recv,
                }
            except:
                network_stats = {"error": "Network stats unavailable"}

            return {
                "cpu_percent": cpu_percent,
                "memory": {
                    "total": memory.total,
                    "available": memory.available,
                    "percent": memory.percent,
                    "used": memory.used,
                },
                "swap": {
                    "total": swap.total,
                    "used": swap.used,
                    "percent": swap.percent,
                },
                "disk": {
                    "total": disk_usage.total,
                    "used": disk_usage.used,
                    "free": disk_usage.free,
                    "percent": (disk_usage.used / disk_usage.total) * 100,
                },
                "network": network_stats,
                "process_count": len(list(psutil.process_iter())),
                "boot_time": psutil.boot_time(),
            }
        except Exception as e:
            self.logger.error(f"Error getting common metrics: {e}")
            return {"error": str(e)}


class LinuxProvider(BasePlatformProvider):
    """Linux-specific implementation"""

    async def get_platform_info(self) -> PlatformInfo:
        """Get Linux-specific platform information"""
        try:
            # Try to get distribution info
            distro_info = await self._get_distro_info()
            package_manager = await self._detect_package_manager()
            desktop_env = await self._detect_desktop_environment()

            return PlatformInfo(
                platform_type=PlatformType.LINUX,
                os_name=distro_info.get("name", "Linux"),
                os_version=distro_info.get("version", platform.release()),
                architecture=platform.architecture()[0],
                kernel_version=platform.release(),
                python_version=sys.version,
                has_admin_rights=await self.check_admin_privileges(),
                package_manager=package_manager,
                shell=await self._get_shell(),
                desktop_environment=desktop_env,
            )
        except Exception as e:
            self.logger.error(f"Error getting Linux platform info: {e}")
            # Fallback to basic info
            return PlatformInfo(
                platform_type=PlatformType.LINUX,
                os_name="Linux",
                os_version=platform.release(),
                architecture=platform.architecture()[0],
                kernel_version=platform.release(),
                python_version=sys.version,
                has_admin_rights=False,
            )

    async def get_system_metrics(self) -> SystemMetrics:
        """Get Linux-specific system metrics"""
        common_metrics = await self.get_common_metrics()

        # Linux-specific additions
        load_avg = None
        temperature = None

        try:
            # Load average (Linux/Unix specific)
            import os

            if hasattr(os, "getloadavg"):
                load_avg = list(os.getloadavg())
        except:
            pass

        try:
            # CPU temperature (Linux specific)
            if hasattr(psutil, "sensors_temperatures"):
                temps = psutil.sensors_temperatures()
                if temps:
                    temperature = {}
                    for name, entries in temps.items():
                        temperature[name] = [
                            {"label": entry.label or "temp", "current": entry.current}
                            for entry in entries
                        ]
        except:
            pass

        return SystemMetrics(
            cpu_usage=common_metrics.get("cpu_percent", 0),
            memory_usage=common_metrics.get("memory", {}),
            disk_usage=common_metrics.get("disk", {}),
            network_stats=common_metrics.get("network", {}),
            process_count=common_metrics.get("process_count", 0),
            uptime_seconds=psutil.boot_time(),
            load_average=load_avg,
            temperature=temperature,
        )

    async def list_running_services(self) -> List[Dict[str, Any]]:
        """List systemd services on Linux"""
        try:
            # Try systemctl first
            result = await self._run_command(
                [
                    "systemctl",
                    "list-units",
                    "--type=service",
                    "--state=running",
                    "--no-pager",
                ]
            )
            services = []

            if result["success"]:
                lines = result["output"].split("\n")[1:]  # Skip header
                for line in lines:
                    if line.strip() and not line.startswith("●"):
                        parts = line.split()
                        if len(parts) >= 4:
                            services.append(
                                {
                                    "name": parts[0],
                                    "load": parts[1],
                                    "active": parts[2],
                                    "sub": parts[3],
                                    "description": " ".join(parts[4:])
                                    if len(parts) > 4
                                    else "",
                                }
                            )

            return services[:20]  # Limit to first 20 services

        except Exception as e:
            self.logger.error(f"Error listing Linux services: {e}")
            return [{"error": f"Could not list services: {e}"}]

    async def get_network_interfaces(self) -> List[Dict[str, Any]]:
        """Get network interfaces on Linux"""
        interfaces = []
        try:
            # Use psutil for cross-platform compatibility
            addrs = psutil.net_if_addrs()
            stats = psutil.net_if_stats()

            for interface_name, addresses in addrs.items():
                interface_info = {
                    "name": interface_name,
                    "addresses": [],
                    "is_up": stats[interface_name].isup
                    if interface_name in stats
                    else False,
                    "speed": stats[interface_name].speed
                    if interface_name in stats
                    else 0,
                }

                for addr in addresses:
                    interface_info["addresses"].append(
                        {
                            "family": str(addr.family),
                            "address": addr.address,
                            "netmask": addr.netmask,
                            "broadcast": addr.broadcast,
                        }
                    )

                interfaces.append(interface_info)

            return interfaces

        except Exception as e:
            self.logger.error(f"Error getting network interfaces: {e}")
            return [{"error": f"Could not get network interfaces: {e}"}]

    async def check_admin_privileges(self) -> bool:
        """Check if running as root on Linux"""
        import os

        return os.geteuid() == 0

    async def _get_distro_info(self) -> Dict[str, str]:
        """Get Linux distribution information"""
        try:
            # Try /etc/os-release first
            os_release_path = Path("/etc/os-release")
            if os_release_path.exists():
                with open(os_release_path) as f:
                    lines = f.readlines()

                info = {}
                for line in lines:
                    if "=" in line:
                        key, value = line.strip().split("=", 1)
                        info[key.lower()] = value.strip('"')

                return {
                    "name": info.get("name", "Unknown Linux"),
                    "version": info.get("version", "Unknown"),
                    "id": info.get("id", "unknown"),
                }
        except Exception as e:
            self.logger.debug(f"Could not read /etc/os-release: {e}")

        return {"name": "Linux", "version": "Unknown"}

    async def _detect_package_manager(self) -> Optional[str]:
        """Detect the package manager on Linux"""
        managers = [
            ("apt", ["apt", "--version"]),
            ("yum", ["yum", "--version"]),
            ("dnf", ["dnf", "--version"]),
            ("pacman", ["pacman", "--version"]),
            ("zypper", ["zypper", "--version"]),
        ]

        for name, cmd in managers:
            result = await self._run_command(cmd)
            if result["success"]:
                return name

        return None

    async def _detect_desktop_environment(self) -> Optional[str]:
        """Detect desktop environment"""
        import os

        # Check common environment variables
        desktop_vars = ["XDG_CURRENT_DESKTOP", "DESKTOP_SESSION", "GDMSESSION"]

        for var in desktop_vars:
            value = os.environ.get(var)
            if value:
                return value.lower()

        return None

    async def _get_shell(self) -> Optional[str]:
        """Get current shell"""
        import os

        return os.environ.get("SHELL", "/bin/bash").split("/")[-1]

    async def _run_command(self, cmd: List[str]) -> Dict[str, Any]:
        """Run a system command safely"""
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            return {
                "success": result.returncode == 0,
                "output": result.stdout,
                "error": result.stderr,
                "returncode": result.returncode,
            }
        except Exception as e:
            return {"success": False, "output": "", "error": str(e), "returncode": -1}


class MacOSProvider(BasePlatformProvider):
    """macOS-specific implementation"""

    async def get_platform_info(self) -> PlatformInfo:
        """Get macOS-specific platform information"""
        try:
            # macOS version info
            mac_ver = platform.mac_ver()

            return PlatformInfo(
                platform_type=PlatformType.MACOS,
                os_name="macOS",
                os_version=mac_ver[0],
                architecture=platform.architecture()[0],
                kernel_version=platform.release(),
                python_version=sys.version,
                has_admin_rights=await self.check_admin_privileges(),
                package_manager=await self._detect_package_manager(),
                shell=await self._get_shell(),
            )
        except Exception as e:
            self.logger.error(f"Error getting macOS platform info: {e}")
            return PlatformInfo(
                platform_type=PlatformType.MACOS,
                os_name="macOS",
                os_version="Unknown",
                architecture=platform.architecture()[0],
                kernel_version=platform.release(),
                python_version=sys.version,
                has_admin_rights=False,
            )

    async def get_system_metrics(self) -> SystemMetrics:
        """Get macOS-specific system metrics"""
        common_metrics = await self.get_common_metrics()

        # macOS-specific additions
        load_avg = None
        try:
            import os

            if hasattr(os, "getloadavg"):
                load_avg = list(os.getloadavg())
        except:
            pass

        return SystemMetrics(
            cpu_usage=common_metrics.get("cpu_percent", 0),
            memory_usage=common_metrics.get("memory", {}),
            disk_usage=common_metrics.get("disk", {}),
            network_stats=common_metrics.get("network", {}),
            process_count=common_metrics.get("process_count", 0),
            uptime_seconds=psutil.boot_time(),
            load_average=load_avg,
        )

    async def list_running_services(self) -> List[Dict[str, Any]]:
        """List launchd services on macOS"""
        try:
            result = await self._run_command(["launchctl", "list"])
            services = []

            if result["success"]:
                lines = result["output"].split("\n")[1:]  # Skip header
                for line in lines:
                    if line.strip():
                        parts = line.split("\t")
                        if len(parts) >= 3:
                            services.append(
                                {
                                    "pid": parts[0] if parts[0] != "-" else None,
                                    "status": parts[1],
                                    "label": parts[2],
                                }
                            )

            return services[:20]  # Limit to first 20

        except Exception as e:
            self.logger.error(f"Error listing macOS services: {e}")
            return [{"error": f"Could not list services: {e}"}]

    async def get_network_interfaces(self) -> List[Dict[str, Any]]:
        """Get network interfaces on macOS"""
        # Use the same cross-platform approach as Linux
        return await LinuxProvider.get_network_interfaces(self)

    async def check_admin_privileges(self) -> bool:
        """Check if running with admin privileges on macOS"""
        import os

        return os.geteuid() == 0

    async def _detect_package_manager(self) -> Optional[str]:
        """Detect package manager on macOS"""
        managers = [
            ("brew", ["brew", "--version"]),
            ("port", ["port", "version"]),
            ("conda", ["conda", "--version"]),
        ]

        for name, cmd in managers:
            result = await self._run_command(cmd)
            if result["success"]:
                return name

        return None

    async def _get_shell(self) -> Optional[str]:
        """Get current shell on macOS"""
        import os

        return os.environ.get("SHELL", "/bin/zsh").split("/")[-1]

    async def _run_command(self, cmd: List[str]) -> Dict[str, Any]:
        """Run a system command safely on macOS"""
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            return {
                "success": result.returncode == 0,
                "output": result.stdout,
                "error": result.stderr,
                "returncode": result.returncode,
            }
        except Exception as e:
            return {"success": False, "output": "", "error": str(e), "returncode": -1}


class WindowsProvider(BasePlatformProvider):
    """Windows-specific implementation"""

    async def get_platform_info(self) -> PlatformInfo:
        """Get Windows-specific platform information"""
        try:
            win_ver = platform.win32_ver()

            return PlatformInfo(
                platform_type=PlatformType.WINDOWS,
                os_name="Windows",
                os_version=win_ver[0],
                architecture=platform.architecture()[0],
                kernel_version=win_ver[1],
                python_version=sys.version,
                has_admin_rights=await self.check_admin_privileges(),
                package_manager=await self._detect_package_manager(),
                shell=await self._get_shell(),
            )
        except Exception as e:
            self.logger.error(f"Error getting Windows platform info: {e}")
            return PlatformInfo(
                platform_type=PlatformType.WINDOWS,
                os_name="Windows",
                os_version="Unknown",
                architecture=platform.architecture()[0],
                kernel_version="Unknown",
                python_version=sys.version,
                has_admin_rights=False,
            )

    async def get_system_metrics(self) -> SystemMetrics:
        """Get Windows-specific system metrics"""
        common_metrics = await self.get_common_metrics()

        return SystemMetrics(
            cpu_usage=common_metrics.get("cpu_percent", 0),
            memory_usage=common_metrics.get("memory", {}),
            disk_usage=common_metrics.get("disk", {}),
            network_stats=common_metrics.get("network", {}),
            process_count=common_metrics.get("process_count", 0),
            uptime_seconds=psutil.boot_time(),
        )

    async def list_running_services(self) -> List[Dict[str, Any]]:
        """List Windows services"""
        try:
            # Use PowerShell to get service information
            result = await self._run_command(
                [
                    "powershell",
                    "-Command",
                    "Get-Service | Where-Object {$_.Status -eq 'Running'} | Select-Object Name, Status, DisplayName | ConvertTo-Json",
                ]
            )

            if result["success"] and result["output"]:
                services_data = json.loads(result["output"])
                if isinstance(services_data, list):
                    return services_data[:20]  # Limit to first 20
                else:
                    return [services_data]  # Single service case

            return []

        except Exception as e:
            self.logger.error(f"Error listing Windows services: {e}")
            return [{"error": f"Could not list services: {e}"}]

    async def get_network_interfaces(self) -> List[Dict[str, Any]]:
        """Get network interfaces on Windows"""
        # Use the same cross-platform approach
        return await LinuxProvider.get_network_interfaces(self)

    async def check_admin_privileges(self) -> bool:
        """Check if running as Administrator on Windows"""
        try:
            import ctypes

            return ctypes.windll.shell32.IsUserAnAdmin() != 0
        except:
            return False

    async def _detect_package_manager(self) -> Optional[str]:
        """Detect package manager on Windows"""
        managers = [
            ("chocolatey", ["choco", "--version"]),
            ("winget", ["winget", "--version"]),
            ("conda", ["conda", "--version"]),
            ("pip", ["pip", "--version"]),
        ]

        for name, cmd in managers:
            result = await self._run_command(cmd)
            if result["success"]:
                return name

        return None

    async def _get_shell(self) -> Optional[str]:
        """Get current shell on Windows"""
        import os

        comspec = os.environ.get("COMSPEC", "cmd.exe")
        return comspec.split("\\")[-1]

    async def _run_command(self, cmd: List[str]) -> Dict[str, Any]:
        """Run a system command safely on Windows"""
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=15,  # Longer timeout for Windows
                shell=True,  # Required for PowerShell commands
            )
            return {
                "success": result.returncode == 0,
                "output": result.stdout,
                "error": result.stderr,
                "returncode": result.returncode,
            }
        except Exception as e:
            return {"success": False, "output": "", "error": str(e), "returncode": -1}


class PlatformManager:
    """Manages platform detection and provider selection"""

    def __init__(self):
        self.current_platform = self._detect_platform()
        self.provider = self._create_provider()
        self.logger = logging.getLogger("PlatformManager")

    def _detect_platform(self) -> PlatformType:
        """Detect the current platform"""
        system = platform.system().lower()

        if system == "linux":
            return PlatformType.LINUX
        elif system == "darwin":
            return PlatformType.MACOS
        elif system == "windows":
            return PlatformType.WINDOWS
        else:
            return PlatformType.UNKNOWN

    def _create_provider(self) -> BasePlatformProvider:
        """Create the appropriate platform provider"""
        if self.current_platform == PlatformType.LINUX:
            return LinuxProvider()
        elif self.current_platform == PlatformType.MACOS:
            return MacOSProvider()
        elif self.current_platform == PlatformType.WINDOWS:
            return WindowsProvider()
        else:
            # Fallback to Linux provider for unknown platforms
            self.logger.warning(
                f"Unknown platform {self.current_platform}, using Linux provider"
            )
            return LinuxProvider()

    async def get_platform_info(self) -> PlatformInfo:
        """Get platform information using the appropriate provider"""
        return await self.provider.get_platform_info()

    async def get_system_metrics(self) -> SystemMetrics:
        """Get system metrics using the appropriate provider"""
        return await self.provider.get_system_metrics()

    async def list_running_services(self) -> List[Dict[str, Any]]:
        """List running services using the appropriate provider"""
        return await self.provider.list_running_services()

    async def get_network_interfaces(self) -> List[Dict[str, Any]]:
        """Get network interfaces using the appropriate provider"""
        return await self.provider.get_network_interfaces()


# FastMCP Server Implementation
# =============================

# Initialize FastMCP
mcp = FastMCP("Platform-Aware System Monitor")

# Global platform manager
platform_manager = PlatformManager()


@mcp.tool()
async def get_platform_info(include_sensitive: bool = False) -> str:
    """
    Get comprehensive platform information

    Args:
        include_sensitive: Whether to include potentially sensitive system info

    Returns:
        JSON string with platform information
    """
    try:
        platform_info = await platform_manager.get_platform_info()
        result = asdict(platform_info)

        # Add current platform detection info
        result["detected_platform"] = platform_manager.current_platform.value
        result["provider_type"] = platform_manager.provider.__class__.__name__

        # Optionally redact sensitive information
        if not include_sensitive:
            # Remove potentially sensitive paths from Python version
            result["python_version"] = result["python_version"].split()[0]

        return json.dumps(result, indent=2)

    except Exception as e:
        return json.dumps({"error": f"Failed to get platform info: {str(e)}"}, indent=2)


@mcp.tool()
async def get_system_metrics(detailed: bool = False) -> str:
    """
    Get current system metrics optimized for the current platform

    Args:
        detailed: Whether to include detailed platform-specific metrics

    Returns:
        JSON string with system metrics
    """
    try:
        metrics = await platform_manager.get_system_metrics()
        result = asdict(metrics)

        # Add metadata
        result["platform"] = platform_manager.current_platform.value
        result["timestamp"] = psutil.boot_time()

        # Include detailed info if requested
        if detailed:
            result["platform_provider"] = platform_manager.provider.__class__.__name__
            result[
                "has_admin_rights"
            ] = await platform_manager.provider.check_admin_privileges()

        return json.dumps(result, indent=2)

    except Exception as e:
        return json.dumps(
            {"error": f"Failed to get system metrics: {str(e)}"}, indent=2
        )


@mcp.tool()
async def list_system_services(limit: int = 20) -> str:
    """
    List running system services (implementation varies by platform)

    Args:
        limit: Maximum number of services to return

    Returns:
        JSON string with service information
    """
    try:
        services = await platform_manager.list_running_services()

        # Limit results
        limited_services = services[:limit]

        result = {
            "platform": platform_manager.current_platform.value,
            "service_count": len(limited_services),
            "total_available": len(services),
            "services": limited_services,
            "note": f"Service listing method varies by platform ({platform_manager.current_platform.value})",
        }

        return json.dumps(result, indent=2)

    except Exception as e:
        return json.dumps({"error": f"Failed to list services: {str(e)}"}, indent=2)


@mcp.tool()
async def get_network_info() -> str:
    """
    Get network interface information

    Returns:
        JSON string with network interface details
    """
    try:
        interfaces = await platform_manager.get_network_interfaces()

        result = {
            "platform": platform_manager.current_platform.value,
            "interface_count": len(interfaces),
            "interfaces": interfaces,
        }

        return json.dumps(result, indent=2)

    except Exception as e:
        return json.dumps({"error": f"Failed to get network info: {str(e)}"}, indent=2)


@mcp.tool()
async def compare_platforms() -> str:
    """
    Compare platform capabilities and show what's available on current platform

    Returns:
        JSON string with platform comparison
    """
    current_platform = platform_manager.current_platform

    # Platform capability matrix
    capabilities = {
        PlatformType.LINUX: {
            "service_manager": "systemd",
            "package_managers": ["apt", "yum", "dnf", "pacman"],
            "load_average": True,
            "temperature_sensors": True,
            "desktop_environments": True,
            "admin_detection": "root/sudo",
            "shell_detection": True,
        },
        PlatformType.MACOS: {
            "service_manager": "launchd",
            "package_managers": ["brew", "port"],
            "load_average": True,
            "temperature_sensors": False,
            "desktop_environments": False,
            "admin_detection": "root/sudo",
            "shell_detection": True,
        },
        PlatformType.WINDOWS: {
            "service_manager": "Windows Services",
            "package_managers": ["chocolatey", "winget"],
            "load_average": False,
            "temperature_sensors": False,
            "desktop_environments": False,
            "admin_detection": "Administrator",
            "shell_detection": True,
        },
    }

    result = {
        "current_platform": current_platform.value,
        "current_capabilities": capabilities.get(current_platform, {}),
        "all_platforms": capabilities,
        "notes": {
            "linux": "Most comprehensive monitoring capabilities",
            "macos": "Similar to Linux but limited temperature monitoring",
            "windows": "Service management different, no load average",
        },
    }

    return json.dumps(result, indent=2)


# Demo and Testing Functions
# ==========================


async def demo_platform_detection():
    """Demonstrate platform detection and capabilities"""
    print("=== Platform Detection Demo ===")

    print(f"\nDetected Platform: {platform_manager.current_platform.value}")
    print(f"Using Provider: {platform_manager.provider.__class__.__name__}")

    # Test platform info
    print("\n1. Platform Information:")
    platform_info = await platform_manager.get_platform_info()
    print(f"   OS: {platform_info.os_name} {platform_info.os_version}")
    print(f"   Architecture: {platform_info.architecture}")
    print(f"   Admin Rights: {platform_info.has_admin_rights}")
    print(f"   Package Manager: {platform_info.package_manager or 'None detected'}")

    # Test system metrics
    print("\n2. System Metrics:")
    metrics = await platform_manager.get_system_metrics()
    print(f"   CPU Usage: {metrics.cpu_usage:.1f}%")
    print(f"   Memory Usage: {metrics.memory_usage.get('percent', 0):.1f}%")
    print(f"   Process Count: {metrics.process_count}")
    if metrics.load_average:
        print(f"   Load Average: {metrics.load_average}")
    if metrics.temperature:
        print(f"   Temperature Sensors: {len(metrics.temperature)} groups")

    # Test services
    print("\n3. Running Services (first 3):")
    services = await platform_manager.list_running_services()
    for i, service in enumerate(services[:3]):
        if "error" not in service:
            name = service.get("name") or service.get("label", "Unknown")
            print(f"   - {name}")
        else:
            print(f"   Error: {service['error']}")

    # Test network interfaces
    print("\n4. Network Interfaces:")
    interfaces = await platform_manager.get_network_interfaces()
    for interface in interfaces[:2]:  # Show first 2
        if "error" not in interface:
            print(
                f"   - {interface['name']}: {'UP' if interface.get('is_up') else 'DOWN'}"
            )
        else:
            print(f"   Error: {interface['error']}")


async def test_fastmcp_tools():
    """Test all FastMCP tools"""
    print("\n=== FastMCP Tools Test ===")

    # Test platform info tool
    print("\n1. Testing get_platform_info tool:")
    result1 = await get_platform_info(include_sensitive=False)
    data1 = json.loads(result1)
    print(f"   Platform: {data1.get('os_name')} {data1.get('os_version')}")
    print(f"   Provider: {data1.get('provider_type')}")

    # Test system metrics tool
    print("\n2. Testing get_system_metrics tool:")
    result2 = await get_system_metrics(detailed=True)
    data2 = json.loads(result2)
    print(f"   CPU: {data2.get('cpu_usage', 0):.1f}%")
    print(f"   Memory: {data2.get('memory_usage', {}).get('percent', 0):.1f}%")

    # Test services tool
    print("\n3. Testing list_system_services tool:")
    result3 = await list_system_services(limit=5)
    data3 = json.loads(result3)
    print(f"   Found: {data3.get('service_count', 0)} services")

    # Test network tool
    print("\n4. Testing get_network_info tool:")
    result4 = await get_network_info()
    data4 = json.loads(result4)
    print(f"   Interfaces: {data4.get('interface_count', 0)}")

    # Test comparison tool
    print("\n5. Testing compare_platforms tool:")
    result5 = await compare_platforms()
    data5 = json.loads(result5)
    current_caps = data5.get("current_capabilities", {})
    print(f"   Service Manager: {current_caps.get('service_manager', 'Unknown')}")
    print(f"   Load Average Support: {current_caps.get('load_average', False)}")


def demonstrate_platform_patterns():
    """Demonstrate key platform abstraction patterns"""
    print("\n=== Platform Abstraction Patterns ===")

    print("\n1. Factory Pattern:")
    print("   - PlatformManager detects OS and creates appropriate provider")
    print("   - Each provider implements the same interface")
    print("   - Automatic fallback for unknown platforms")

    print("\n2. Protocol-Based Design:")
    print("   - PlatformProvider protocol defines common interface")
    print("   - Each OS implementation can vary internally")
    print("   - Type safety with Python typing")

    print("\n3. Graceful Degradation:")
    print("   - Common metrics work on all platforms")
    print("   - Platform-specific features fail safely")
    print("   - Error handling preserves functionality")

    print("\n4. Performance Optimization:")
    print("   - Platform-specific commands for efficiency")
    print("   - Caching strategies per platform")
    print("   - Timeout handling for slow operations")


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Run as MCP server
    mcp.run()


# pyproject.toml - UV Project Configuration
"""
[project]
name = "mcp-tutorial-modern"
version = "0.1.0"
description = "Modern MCP Server Tutorial using UV and FastMCP"
dependencies = [
    "fastmcp>=0.9.0",
    "psutil>=5.9.0",
    "pytest>=7.0.0"
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.uv]
dev-dependencies = [
    "pytest-asyncio>=0.21.0",
    "pytest-cov>=4.0.0",
    "black>=23.0.0",
    "ruff>=0.1.0"
]

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]

[tool.black]
line-length = 100
target-version = ['py38']

[tool.ruff]
line-length = 100
target-version = "py38"
"""

# tests/test_platform_providers.py - Test Suite
"""
import pytest
import platform
from unittest.mock import patch, AsyncMock

from part2_platform_fastmcp import (
    PlatformManager, 
    LinuxProvider, 
    MacOSProvider, 
    WindowsProvider,
    PlatformType
)

class TestPlatformProviders:
    '''Test platform provider implementations'''
    
    @pytest.mark.asyncio
    async def test_linux_provider_info(self):
        '''Test Linux provider platform info'''
        provider = LinuxProvider()
        info = await provider.get_platform_info()
        
        assert info.platform_type == PlatformType.LINUX
        assert isinstance(info.has_admin_rights, bool)
        assert info.architecture is not None
    
    @pytest.mark.asyncio
    async def test_macos_provider_info(self):
        '''Test macOS provider platform info'''
        provider = MacOSProvider()
        info = await provider.get_platform_info()
        
        assert info.platform_type == PlatformType.MACOS
        assert info.os_name == "macOS"
    
    @pytest.mark.asyncio
    async def test_windows_provider_info(self):
        '''Test Windows provider platform info'''
        provider = WindowsProvider()
        info = await provider.get_platform_info()
        
        assert info.platform_type == PlatformType.WINDOWS
        assert info.os_name == "Windows"
    
    @pytest.mark.asyncio 
    async def test_platform_manager_detection(self):
        '''Test platform manager auto-detection'''
        manager = PlatformManager()
        
        # Should detect current platform correctly
        current_os = platform.system().lower()
        if current_os == "linux":
            assert manager.current_platform == PlatformType.LINUX
            assert isinstance(manager.provider, LinuxProvider)
        elif current_os == "darwin":
            assert manager.current_platform == PlatformType.MACOS
            assert isinstance(manager.provider, MacOSProvider)
        elif current_os == "windows":
            assert manager.current_platform == PlatformType.WINDOWS
            assert isinstance(manager.provider, WindowsProvider)
    
    @pytest.mark.asyncio
    async def test_cross_platform_metrics(self):
        '''Test that metrics work across platforms'''
        manager = PlatformManager()
        metrics = await manager.get_system_metrics()
        
        # Basic metrics should always be available
        assert metrics.cpu_usage >= 0
        assert metrics.process_count > 0
        assert isinstance(metrics.memory_usage, dict)
        assert isinstance(metrics.disk_usage, dict)
    
    @pytest.mark.asyncio
    async def test_error_handling(self):
        '''Test error handling in providers'''
        provider = LinuxProvider()
        
        # Mock a failing psutil call
        with patch('psutil.cpu_percent', side_effect=Exception("Test error")):
            common_metrics = await provider.get_common_metrics()
            assert "error" in common_metrics
"""

# Makefile - Development Commands
"""
.PHONY: install test lint format run-demo clean

# Install dependencies
install:
	uv sync

# Run tests
test:
	uv run pytest tests/ -v

# Run linting
lint:
	uv run ruff check .
	uv run black --check .

# Format code
format:
	uv run black .
	uv run ruff --fix .

# Run the demo
run-demo:
	uv run part2_platform_fastmcp.py

# Run all parts
run-all:
	uv run part1_multiple_tools.py
	uv run part2_platform_fastmcp.py

# Clean cache
clean:
	find . -type d -name "__pycache__" -delete
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache
	rm -rf dist/

# Setup development environment
setup-dev: install
	uv add --dev pytest-asyncio pytest-cov black ruff
"""
