# ./src/ollamatoolkit/tools/system.py
"""
Ollama Toolkit - System Tools
=============================
Advanced utilities for deep system introspection, resource monitoring, and hardware auditing.
Supports Windows, Linux, and macOS.
"""

import datetime
import logging
import os
import platform
import shutil
import socket
import sys
import subprocess
import time
from typing import Any, Dict, List, Union

# Optional Psutil for Pro-Grade monitoring
try:
    import psutil
except ImportError:
    psutil = None

logger = logging.getLogger(__name__)


class SystemTools:
    """
    Advanced system introspection and process control.
    """

    @staticmethod
    def get_current_time(timezone: str = "local") -> str:
        """Returns the current date and time in ISO format."""
        if timezone.lower() == "utc":
            now = datetime.datetime.now(datetime.timezone.utc)
        else:
            now = datetime.datetime.now()
        return now.isoformat(timespec="seconds")

    @staticmethod
    def wait_seconds(seconds: int) -> str:
        """Pauses execution (max 60s)."""
        seconds = max(0, min(seconds, 60))
        time.sleep(seconds)
        return f"Waited {seconds} seconds."

    @staticmethod
    def get_system_report(full: bool = True) -> Dict[str, Any]:
        """
        Returns a comprehensive report of the system state.
        :param full: If True, includes slow calls like installed software.
        """
        report: Dict[str, Any] = {
            "timestamp": SystemTools.get_current_time(),
            "os_info": SystemTools._get_os_info(),
            "hardware": SystemTools._get_hardware_info(),
            "network": SystemTools._get_network_info(),
            "environment": SystemTools._get_python_info(),
        }

        if full:
            report["installed_software"] = SystemTools._get_installed_software()
            report["gpu_info"] = SystemTools._get_gpu_info()

        return report

    @staticmethod
    def monitor_resources(
        target_process: str = "ollama", duration_sec: int = 1
    ) -> Dict[str, Any]:
        """
        Monitors system load and specific process usage.
        :param target_process: Name of process to search for (partial match).
        """
        if not psutil:
            return {"error": "psutil not installed. Cannot monitor resources."}

        stats = {
            "timestamp": SystemTools.get_current_time(),
            "cpu_percent_total": psutil.cpu_percent(interval=duration_sec),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_io": str(psutil.disk_io_counters() or "N/A"),
            "net_io": str(psutil.net_io_counters() or "N/A"),
            "targets": [],
        }

        # Find target process
        for proc in psutil.process_iter(
            ["pid", "name", "cpu_percent", "memory_percent"]
        ):
            try:
                if target_process.lower() in proc.info["name"].lower():
                    stats["targets"].append(
                        {
                            "pid": proc.info["pid"],
                            "name": proc.info["name"],
                            "cpu_usage": proc.info["cpu_percent"],
                            "mem_usage": round(proc.info["memory_percent"], 2),
                        }
                    )
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        return stats

    # --- Internal Helpers ---

    @staticmethod
    def _get_os_info() -> Dict[str, str]:
        return {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "architecture": platform.machine(),
            "node_name": platform.node(),
        }

    @staticmethod
    def _get_hardware_info() -> Dict[str, Any]:
        info: Dict[str, Any] = {
            "cpu": {
                "logical_cores": os.cpu_count() or 1,
                "arch": platform.machine(),
                "freq": "N/A",
            },
            "memory": {"total_gb": "N/A", "available_gb": "N/A"},
            "disk": SystemTools._get_disk_info(),
        }

        if psutil:
            info["cpu"]["logical_cores"] = psutil.cpu_count(logical=True)
            info["cpu"]["physical_cores"] = psutil.cpu_count(logical=False)
            try:
                freq = psutil.cpu_freq()
                if freq:
                    info["cpu"]["freq_mhz"] = freq.current
            except Exception:
                pass

            mem = psutil.virtual_memory()
            info["memory"] = {
                "total_gb": round(mem.total / (1024**3), 2),
                "available_gb": round(mem.available / (1024**3), 2),
                "percent": mem.percent,
            }

        # Motherboard (Advanced)
        sys_plat = platform.system()
        try:
            if sys_plat == "Windows":
                # WMI Baseboard
                cmd = "wmic baseboard get product,manufacturer,version,serialnumber /format:list"
                res = subprocess.check_output(cmd, shell=True, text=True)
                info["motherboard"] = {
                    k.strip(): v.strip()
                    for k, v in [
                        line.split("=", 1)
                        for line in res.strip().split("\n")
                        if "=" in line
                    ]
                }
            elif sys_plat == "Linux":
                # Dmidecode (usually requires sudo, might accept 'wont fix' or try /sys/class/dmi/id)
                # Fallback to readable files
                info["motherboard"] = "Root/Sudo required for DMI data on Linux"
        except Exception as e:
            info["motherboard_error"] = str(e)

        return info

    @staticmethod
    def _get_disk_info() -> List[Dict]:
        """Returns details for all mounted partitions."""
        disks = []
        if psutil:
            for part in psutil.disk_partitions(all=False):
                try:
                    usage = psutil.disk_usage(part.mountpoint)
                    disks.append(
                        {
                            "device": part.device,
                            "mountpoint": part.mountpoint,
                            "fstype": part.fstype,
                            "total_gb": round(usage.total / (1024**3), 2),
                            "free_gb": round(usage.free / (1024**3), 2),
                            "percent": usage.percent,
                        }
                    )
                except Exception:
                    pass
        else:
            # Fallback
            total, used, free = shutil.disk_usage(".")
            disks.append(
                {
                    "mountpoint": ".",
                    "total_gb": round(total / (1024**3), 2),
                    "free_gb": round(free / (1024**3), 2),
                }
            )
        return disks

    @staticmethod
    def _get_network_info() -> Dict[str, Any]:
        net: Dict[str, Any] = {
            "hostname": socket.gethostname(),
            "ip_local": "Unknown",
            "interfaces": [],
            "proxies": {
                "http_proxy": os.environ.get("http_proxy"),
                "https_proxy": os.environ.get("https_proxy"),
                "no_proxy": os.environ.get("no_proxy"),
            },
        }

        # Local IP Trick
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("10.254.254.254", 1))
            net["ip_local"] = s.getsockname()[0]
            s.close()
        except Exception:
            pass

        if psutil:
            addrs = psutil.net_if_addrs()
            for name, snics in addrs.items():
                ips = []
                for snic in snics:
                    if snic.family == socket.AF_INET:
                        ips.append(snic.address)
                if ips:
                    net["interfaces"].append({"name": name, "ipv4": ips})

        return net

    @staticmethod
    def _get_installed_software() -> Union[List[str], str]:
        """
        Attempts to list installed software.
        """
        sys_plat = platform.system()
        try:
            if sys_plat == "Windows":
                # Try PowerShell (Registry) - much faster than WMIC
                cmd = 'powershell "Get-ItemProperty HKLM:\\Software\\Wow6432Node\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\* | Select-Object DisplayName"'
                try:
                    res = subprocess.check_output(cmd, shell=True, text=True)
                    apps = [line.strip() for line in res.split("\n") if line.strip()]
                    return apps[:50] + (["..."] if len(apps) > 50 else [])
                except Exception as e:
                    return f"PowerShell query failed: {e}. (Wmic is too slow to use safely)."

            elif sys_plat == "Linux":
                # Try dpkg
                try:
                    res = subprocess.check_output(
                        "dpkg-query -f '${Package}\n' -W", shell=True, text=True
                    )
                    apps = res.strip().split("\n")
                    return apps[:50] + (["..."] if len(apps) > 50 else [])
                except Exception:
                    return "Could not list software (requires dpkg/rpm)."
            elif sys_plat == "Darwin":
                return "Mac software list requires system_profiler (slow)."
        except Exception as e:
            return f"Error: {e}"
        return "Unknown OS"

    @staticmethod
    def _get_gpu_info() -> Union[List[Dict[str, str]], str]:
        """
        Attempts to fetch GPU info via nvidia-smi.
        """
        try:
            # Try NVIDIA
            res = subprocess.check_output(
                "nvidia-smi --query-gpu=name,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader",
                shell=True,
                text=True,
            )
            gpus: List[Dict[str, str]] = []
            for line in res.strip().split("\n"):
                if line.strip():
                    parts = line.split(",")
                    gpus.append(
                        {
                            "name": parts[0].strip(),
                            "memory_total": parts[1].strip(),
                            "load_percent": parts[2].strip(),
                            "temp_c": parts[3].strip(),
                        }
                    )
            return gpus
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass

        # Fallback Windows
        if platform.system() == "Windows":
            try:
                cmd = "wmic path win32_videocontroller get name,adapterram /format:list"
                res = subprocess.check_output(cmd, shell=True, text=True)
                entries: List[Dict[str, str]] = []
                for line in res.strip().split("\n"):
                    if "=" not in line:
                        continue
                    key, value = line.split("=", 1)
                    entries.append({key.strip(): value.strip()})
                return entries
            except Exception:
                pass

        return "No GPU Detected / Drivers missing."

    @staticmethod
    def _get_python_info() -> Dict[str, Any]:
        return {
            "version": platform.python_version(),
            "executable": sys.executable,
            "venv": sys.prefix != sys.base_prefix,
        }
