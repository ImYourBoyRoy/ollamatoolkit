# ./src/ollamatoolkit/dashboard.py
"""
Ollama Toolkit - Mission Control Dashboard
==========================================
Terminal User Interface (TUI) for real-time agent monitoring.
Powered by Rich.
"""

import queue
import threading
import time

try:
    import importlib.util

    if (
        importlib.util.find_spec("rich") is None
        or importlib.util.find_spec("rich.panel") is None
        or importlib.util.find_spec("rich.live") is None
        or importlib.util.find_spec("rich.layout") is None
        or importlib.util.find_spec("rich.text") is None
        or importlib.util.find_spec("rich.console") is None
        or importlib.util.find_spec("rich.table") is None
        or importlib.util.find_spec("rich.ansi") is None
    ):
        raise ImportError("Rich not fully installed")

    from rich.panel import Panel
    from rich.text import Text

    # from rich.console import Group
    # from rich.table import Table
    # from rich.ansi import AnsiDecoder
    # from rich import box
    from rich.live import Live
    from rich.layout import Layout
except ImportError:
    pass  # Handled by check in CLI

from ollamatoolkit.config import DashboardConfig
from ollamatoolkit.tools.system import SystemTools
from ollamatoolkit.agents.simple import SimpleAgent


class DashboardRunner:
    def __init__(self, agent: SimpleAgent, config: DashboardConfig):
        self.agent = agent
        self.config = config
        self.logs_queue: "queue.Queue[str]" = queue.Queue()
        self.tool_status = "Idle"
        self.context_usage = "0 / ?"
        self.last_thought = "Initializing..."
        self.is_running = False

        # Capture Agent Logs (Basic Hooking)
        # In a real heavy implementation, we'd hook the Logger.
        # For now, we rely on the Agent loop updating us or just monitoring system.

    def run(self, task: str):
        """Starts the TUI and the Agent."""
        self.is_running = True

        # Layout
        layout = self._make_layout()

        # Start Agent Thread
        agent_thread = threading.Thread(
            target=self._run_agent, args=(task,), daemon=True
        )
        agent_thread.start()

        # Main UI Loop
        with Live(
            layout, refresh_per_second=1000 // self.config.refresh_rate_ms, screen=True
        ):
            while self.is_running and agent_thread.is_alive():
                self._update_layout(layout)
                time.sleep(self.config.refresh_rate_ms / 1000)

            # One final update
            self._update_layout(layout)

            # Keep open for a moment if done?
            if not self.is_running:  # Explicit exit
                pass
            else:
                # Agent finished
                layout["mind"].update(
                    Panel(
                        Text("Agent Finished Task.", style="bold green"),
                        title="The Mind",
                    )
                )
                time.sleep(2)

    def _run_agent(self, task: str):
        try:
            self.last_thought = f"Received Task: {task}"

            # We hook the agent's 'on_step' if available or just run it.
            # Since SimpleAgent doesn't have deep hooks yet, we just run.
            # Ideally, we'd refactor Agent to emit events.
            # For this "Phase 11", we'll just show System Stats vigorously.

            self.agent.run(task)
            self.last_thought = "Task Completed."
        except Exception as e:
            self.last_thought = f"Error: {e}"
        finally:
            self.is_running = False

    def _make_layout(self) -> Layout:
        layout = Layout()

        # Split: Header, Main (Mind|System), Footer (Tools)
        layout.split(
            Layout(name="header", size=3),
            Layout(name="main", ratio=1),
            Layout(name="footer", size=3),
        )

        layout["main"].split_row(
            Layout(name="mind", ratio=2), Layout(name="system", ratio=1)
        )

        return layout

    def _update_layout(self, layout: Layout):
        # Header
        layout["header"].update(
            Panel(
                Text(
                    f"OLLAMA TOOLKIT | MISSION CONTROL | RUNNING: {self.agent.name}",
                    justify="center",
                    style="bold white",
                ),
                style="on blue",
            )
        )

        # System Panel
        if self.config.show_system:
            sys_report = SystemTools.get_system_report(full=False)  # Fast check

            # Basic Resource Monitor
            # Only if psutil is there, handled by SystemTools fallback
            SystemTools.monitor_resources(duration_sec=0)  # Instant check if possible?
            # monitor_resources sleeps, so we can't call him here in UI loop easily without lag.
            # Better to rely on the static report snapshot or a separate ticker.
            # We'll stick to static snapshot + pseudo realtime from previous calls if we had a monitor thread.
            # For simplicity: Just show HW info.

            sys_text = Text()
            sys_text.append(
                f"OS: {sys_report['os_info']['system']} {sys_report['os_info']['release']}\n",
                style="cyan",
            )
            sys_text.append(
                f"CPU: {sys_report['hardware']['cpu']['logical_cores']} Cores\n"
            )

            # If monitor_resources works (psutil), we'd want that.
            # But it requires blocking measurement. Skip for UI fluidity in this version.

            layout["system"].update(
                Panel(sys_text, title="System Vitality", border_style="cyan")
            )
        else:
            layout["system"].visible = False

        # Mind Panel (Agent State)
        # Since we don't have deep hooks, we show the last status msg.
        layout["mind"].update(
            Panel(
                Text(self.last_thought, style="green"),
                title="The Mind (Stream)",
                border_style="green",
            )
        )

        # Tool Panel
        if self.config.show_tools:
            layout["footer"].update(
                Panel(
                    Text(f"Status: {self.tool_status}", style="yellow"),
                    title="Active Tool",
                    border_style="yellow",
                )
            )
