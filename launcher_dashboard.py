"""
LauncherDashboard: A modern, resizable operational dashboard UI for the 2048 Bot launcher

Goals
- Dynamic, resizable layout with sensible minimums and grid-based scaling
- Operator-focused: surface exact process metrics, health, errors, and controls
- Platform-agnostic visual style using CustomTkinter (dark theme), with ttk Treeview
- Never use approximations if exact values are available
- Clean API so the main launcher can drive updates without tight coupling

Public API
- set callbacks via attributes:
  - on_stop_requested(): stop all services
  - on_restart_requested(): restart all services
  - on_closing(): window close callback
- show(), hide(), destroy()
- update_progress(step: str, progress: float, status: str = "", error: str = "")
- update_status(backend_status: str, frontend_status: str, tunnel_status: str)
- update_process_status(status_dict: dict[str, dict])  # output of ProcessManager.get_process_status()
- append_process_logs(status_dict: dict[str, dict])    # consumes recent_output/errors from status
- show_access_info(frontend_url: str, backend_url: str)
- show_error(error: str), hide_error()

Implementation notes
- Uses ttk.Treeview for a sortable processes table with exact CPU%, Mem%, Uptime (derived via psutil)
- Periodic health checks (true values) for backend (/docs) and frontend (/), with precise latency
- Logs panel with level filters, per-process tags, timestamped entries
- Settings drawer for knobs (disabled if no callback provided by launcher)
"""

from __future__ import annotations

import threading
import time
import math
from datetime import datetime
from typing import Any, Dict, Optional
from pathlib import Path
import json
import re

import platform
import socket

try:
    import customtkinter as ctk
    import tkinter as tk
    from tkinter import ttk
    from tkinter import font as tkfont
except Exception as e:  # pragma: no cover
    raise RuntimeError(f"CustomTkinter is required for LauncherDashboard: {e}")

try:
    import psutil
except Exception as e:  # pragma: no cover
    psutil = None

try:
    import requests
except Exception as e:  # pragma: no cover
    requests = None

try:
    import qrcode
except Exception:
    qrcode = None
try:
    from PIL import Image, ImageTk
except Exception:
    Image = None
    ImageTk = None


class LauncherDashboard:
    """Operational dashboard window for the launcher."""

    # Callbacks to be assigned by the launcher
    on_stop_requested: Optional[callable] = None
    on_restart_requested: Optional[callable] = None
    on_closing: Optional[callable] = None
    on_process_action: Optional[callable] = None  # (name: str, action: str) -> None

    def __init__(self, logger: Any):
        self.logger = logger
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.window = ctk.CTk()
        self.window.title("2048 Bot — Launcher Dashboard")
        self.window.geometry("1100x720")
        self.window.minsize(980, 620)
        # Allow resize
        self.window.resizable(True, True)

        # Set app icon if available
        try:
            icon_path = Path("project_icon.png")
            if icon_path.exists():
                if platform.system() == "Windows":
                    from PIL import Image
                    ico_path = Path("project_icon.ico")
                    if not ico_path.exists():
                        Image.open(icon_path).save(ico_path, format='ICO')
                    self.window.iconbitmap(str(ico_path))
                else:
                    self.window.iconphoto(True, tk.PhotoImage(file=str(icon_path)))
        except Exception:
            pass

        # Tracking
        self._process_rows: Dict[str, Dict[str, Any]] = {}
        self._pid_cache: Dict[str, Optional[int]] = {}
        self._health_targets = {"backend": None, "frontend": None}  # URLs
        self._health_stats = {
            "backend": {"status": "Unknown", "latency_ms": None, "last": None},
            "frontend": {"status": "Unknown", "latency_ms": None, "last": None},
        }
        self._stop_event = threading.Event()
        self._prefs_path = Path("launcher_ui_prefs.json")
        self._prefs = {}
        self._prefs_snapshot = {}

        # Grid layout: header, body (paned), logs, footer
        self.window.grid_rowconfigure(1, weight=1)
        self.window.grid_rowconfigure(2, weight=1)
        self.window.grid_rowconfigure(3, weight=0)
        self.window.grid_columnconfigure(0, weight=1)

        self._build_header()
        self._build_body()
        self._build_logs()
        self._load_prefs_and_apply()

        # Events
        self.window.protocol("WM_DELETE_WINDOW", self._on_window_close)

        # Background tasks
        self._start_periodic_health_checks()
        self._start_progress_animation()

    # ---------- UI Build ----------
    def _build_header(self):
        header = ctk.CTkFrame(self.window)
        header.grid(row=0, column=0, sticky="nsew", padx=20, pady=(20, 14))
        header.grid_columnconfigure(0, weight=1)

        # Title and status chips
        title_row = ctk.CTkFrame(header, fg_color="transparent")
        title_row.grid(row=0, column=0, sticky="ew", padx=12, pady=(6, 10))
        title_row.grid_columnconfigure(0, weight=1)

        title = ctk.CTkLabel(title_row, text="2048 Bot Launcher Dashboard", font=ctk.CTkFont(size=20, weight="bold"))
        title.grid(row=0, column=0, sticky="w", padx=(2, 12))

        btn_row = ctk.CTkFrame(title_row, fg_color="transparent")
        btn_row.grid(row=0, column=1, sticky="e")

        self.btn_stop = ctk.CTkButton(btn_row, text="Stop All", fg_color="#ef4444", hover_color="#dc2626", command=self._cb_stop)
        self.btn_stop.grid(row=0, column=0, padx=(0, 12), pady=2)
        self.btn_restart = ctk.CTkButton(btn_row, text="Restart", command=self._cb_restart)
        self.btn_restart.grid(row=0, column=1, padx=(0, 2), pady=2)

        # Status chips
        chips = ctk.CTkFrame(header)
        chips.grid(row=1, column=0, sticky="ew", pady=(12, 10), padx=12)
        for i in range(6):
            chips.grid_columnconfigure(i, weight=1)

        self._chip_backend = self._make_chip(chips, 0, "Backend", "Unknown")
        self._chip_frontend = self._make_chip(chips, 1, "Frontend", "Unknown")
        self._chip_tunnel = self._make_chip(chips, 2, "Tunnel", "Not started")
        self._chip_ip = self._make_chip(chips, 3, "Host", socket.gethostname())
        self._chip_backend_latency = self._make_chip(chips, 4, "API Latency", "—")
        self._chip_frontend_latency = self._make_chip(chips, 5, "UI Latency", "—")

        # Progress + error line
        prog_row = ctk.CTkFrame(header, fg_color="transparent")
        prog_row.grid(row=2, column=0, sticky="ew", pady=(16, 8), padx=12)
        prog_row.grid_columnconfigure(1, weight=1)

        self.lbl_step = ctk.CTkLabel(prog_row, text="Initializing…", font=ctk.CTkFont(size=13, weight="bold"))
        self.lbl_step.grid(row=0, column=0, padx=(0, 14))
        # Smooth progress widget (replaces glitchy default)
        self.progress = _SmoothProgress(prog_row, height=16)
        self.progress.grid(row=0, column=1, sticky="ew", padx=(0, 6))

        self.lbl_error = ctk.CTkLabel(header, text="", text_color="#f87171", font=ctk.CTkFont(size=12))
        self.lbl_error.grid(row=3, column=0, sticky="w", pady=(10, 2), padx=12)

    def _make_chip(self, parent: ctk.CTkFrame, col: int, label: str, value: str):
        frame = ctk.CTkFrame(parent, corner_radius=12, border_width=1, border_color="#374151", fg_color="#0f172a")
        frame.grid(row=0, column=col, sticky="ew", padx=8, pady=2)
        frame.grid_columnconfigure(2, weight=1)
        dot = ctk.CTkLabel(frame, text="●", text_color="#6b7280", font=ctk.CTkFont(size=18, weight="bold"))
        dot.grid(row=0, column=0, padx=10, pady=10, sticky="w")
        name_lbl = ctk.CTkLabel(frame, text=label, text_color="#9ca3af", font=ctk.CTkFont(size=11, weight="bold"))
        name_lbl.grid(row=0, column=1, padx=(0, 8), pady=10, sticky="w")
        value_lbl = ctk.CTkLabel(frame, text=value, font=ctk.CTkFont(size=12, weight="bold"))
        value_lbl.grid(row=0, column=2, padx=(0, 12), pady=10, sticky="w")
        chip = {"frame": frame, "dot": dot, "name_label": name_lbl, "value": value_lbl, "label": value_lbl, "name": label, "full_value_text": value}
        try:
            frame.bind("<Configure>", lambda e, ch=chip: self._reflow_chip(ch))
            value_lbl.bind("<Configure>", lambda e, ch=chip: self._reflow_chip(ch))
        except Exception:
            pass
        return chip

    def _build_body(self):
        # Paned body for resizable split
        body = tk.PanedWindow(self.window, orient='horizontal', sashwidth=6, bg='#0b1220', bd=0, relief='flat')
        body.grid(row=1, column=0, sticky="nsew", padx=20, pady=(0, 6))

        # Processes panel
        proc_panel = ctk.CTkFrame(body)
        proc_panel.grid_rowconfigure(1, weight=1)
        proc_panel.grid_columnconfigure(0, weight=1)
        body.add(proc_panel, minsize=420, stretch='always')

        # Processes header with action bar
        header_row = ctk.CTkFrame(proc_panel, fg_color="transparent")
        header_row.grid(row=0, column=0, sticky="ew", padx=14, pady=(16, 10))
        header_row.grid_columnconfigure(0, weight=1)
        header = ctk.CTkLabel(header_row, text="Processes", font=ctk.CTkFont(size=14, weight="bold"))
        header.grid(row=0, column=0, sticky="w", padx=6)
        self.proc_filter_var = tk.StringVar()
        self.proc_filter = ctk.CTkEntry(header_row, placeholder_text="Filter processes (regex)", textvariable=self.proc_filter_var, width=220)
        self.proc_filter.grid(row=0, column=1, sticky="e", padx=(8, 2))
        self.proc_filter_var.trace_add('write', lambda *_: self._apply_process_filter())
        # Compact actions
        actions_bar = ctk.CTkFrame(proc_panel, fg_color="transparent")
        actions_bar.grid(row=2, column=0, sticky="ew", padx=14, pady=(2, 12))
        actions_bar.grid_columnconfigure(0, weight=1)
        btn_term = ctk.CTkButton(actions_bar, text="Terminate", width=90, command=lambda: self._proc_action('terminate'))
        btn_kill = ctk.CTkButton(actions_bar, text="Kill", width=70, fg_color="#ef4444", hover_color="#dc2626", command=lambda: self._proc_action('kill'))
        btn_restart = ctk.CTkButton(actions_bar, text="Restart", width=90, command=lambda: self._proc_action('restart'))
        btn_term.grid(row=0, column=1, padx=(0, 10), pady=2)
        btn_kill.grid(row=0, column=2, padx=(0, 10), pady=2)
        btn_restart.grid(row=0, column=3, padx=(0, 2), pady=2)

        # Treeview for process table (ttk inside CTk)
        # Force dark theme by using a custom style to avoid white header/areas
        style = ttk.Style()
        try:
            style.theme_use('clam')
        except Exception:
            pass
        style.configure(
            "Dark.Treeview",
            background="#0b1220",
            fieldbackground="#0b1220",
            foreground="#d1d5db",
            rowheight=24,
            borderwidth=0
        )
        style.map("Dark.Treeview", background=[('selected', '#1e293b')], foreground=[('selected', '#e5e7eb')])
        style.configure("Dark.Treeview.Heading", background="#0f172a", foreground="#e2e8f0", relief="flat")
        style.map("Dark.Treeview.Heading", background=[('active', '#0f172a')], relief=[('active', 'flat')])

        columns = ("name", "pid", "state", "cpu", "mem", "uptime", "port", "last_error")
        self.proc_tree = ttk.Treeview(proc_panel, columns=columns, show="headings", height=12, style="Dark.Treeview")
        headings = {
            "name": "Name",
            "pid": "PID",
            "state": "State",
            "cpu": "CPU %",
            "mem": "Mem %",
            "uptime": "Uptime",
            "port": "Port",
            "last_error": "Last Error"
        }
        for key, text in headings.items():
            self.proc_tree.heading(key, text=text, command=lambda c=key: self._sort_treeview(c))
            # sensible widths
            self.proc_tree.column(key, width=90 if key != "last_error" else 260, anchor=tk.W, stretch=True)

        # Ensure heading style applied
        try:
            self.proc_tree.heading("name", text="Name")
            style.configure("Dark.Treeview.Heading", padding=4)
        except Exception:
            pass

        vsb = ttk.Scrollbar(proc_panel, orient="vertical", command=self.proc_tree.yview)
        self.proc_tree.configure(yscrollcommand=vsb.set)
        self.proc_tree.grid(row=1, column=0, sticky="nsew", padx=(14, 0), pady=(0, 12))
        vsb.grid(row=1, column=1, sticky="ns", pady=(0, 12), padx=(0, 14))

        # Row striping for readability
        self.proc_tree.tag_configure('odd', background='#0c182b')
        self.proc_tree.tag_configure('even', background='#0b1220')

        # Health / Network / QR panel (right)
        right_panel = ctk.CTkFrame(body)
        right_panel.grid_rowconfigure(0, weight=1)
        right_panel.grid_columnconfigure(0, weight=1)
        body.add(right_panel, minsize=460, stretch='always')
        # Right panel as tabs to prevent QR from crowding health
        self.tabview = ctk.CTkTabview(right_panel, segmented_button_selected_color="#2563eb", segmented_button_unselected_color="#111827")
        self.tabview.grid(row=0, column=0, sticky="nsew", padx=(14, 6), pady=(0, 14))
        tab_overview = self.tabview.add("Overview")
        tab_qr = self.tabview.add("QR")
        tab_overview.grid_rowconfigure(1, weight=1)
        tab_overview.grid_columnconfigure(0, weight=1)
        tab_qr.grid_rowconfigure(0, weight=1)
        tab_qr.grid_columnconfigure(0, weight=1)

        sec_row = ctk.CTkFrame(tab_overview, fg_color="transparent")
        sec_row.grid(row=0, column=0, sticky="ew", padx=12, pady=(12, 10))
        sec_row.grid_columnconfigure(0, weight=1)
        sec_title = ctk.CTkLabel(sec_row, text="Health & Network", font=ctk.CTkFont(size=14, weight="bold"))
        sec_title.grid(row=0, column=0, sticky="w", padx=6)
        self.btn_snapshot = ctk.CTkButton(sec_row, text="Save Snapshot", width=120, command=self._save_health_snapshot)
        self.btn_snapshot.grid(row=0, column=1, sticky="e", padx=(8, 2))

        # Health exact values with actions
        health_box = ctk.CTkFrame(tab_overview)
        health_box.grid(row=1, column=0, sticky="nsew", padx=16, pady=(4, 8))
        for i in range(2):
            health_box.grid_rowconfigure(i, weight=1)
        health_box.grid_columnconfigure(1, weight=1)

        self.lbl_backend_url = ctk.CTkLabel(health_box, text="Backend: —", anchor="w")
        self.lbl_backend_url.grid(row=0, column=0, sticky="ew", padx=12, pady=(12, 2))
        self.lbl_backend_health = ctk.CTkLabel(health_box, text="Status: Unknown | Latency: — | Last: —", anchor="w")
        self.lbl_backend_health.grid(row=0, column=1, sticky="ew", padx=12, pady=(12, 2))

        self.lbl_frontend_url = ctk.CTkLabel(health_box, text="Frontend: —", anchor="w")
        self.lbl_frontend_url.grid(row=1, column=0, sticky="ew", padx=12, pady=(6, 12))
        self.lbl_frontend_health = ctk.CTkLabel(health_box, text="Status: Unknown | Latency: — | Last: —", anchor="w")
        self.lbl_frontend_health.grid(row=1, column=1, sticky="ew", padx=12, pady=(6, 12))

        # Latency sparklines
        spark_row = ctk.CTkFrame(tab_overview, fg_color="transparent")
        spark_row.grid(row=2, column=0, sticky="ew", padx=16, pady=(6, 12))
        spark_row.grid_columnconfigure(0, weight=1)
        spark_row.grid_columnconfigure(1, weight=1)
        self.backend_spark = tk.Canvas(spark_row, height=40, bg="#0f172a", highlightthickness=0)
        self.frontend_spark = tk.Canvas(spark_row, height=40, bg="#0f172a", highlightthickness=0)
        self.backend_spark.grid(row=0, column=0, sticky="ew", padx=(0, 10))
        self.frontend_spark.grid(row=0, column=1, sticky="ew", padx=(0, 0))
        self._latency_history = {"backend": [], "frontend": []}

        # Action buttons
        actions = ctk.CTkFrame(tab_overview, fg_color="transparent")
        actions.grid(row=3, column=0, sticky="ew", padx=16, pady=(4, 14))
        actions.grid_columnconfigure(0, weight=1)
        actions.grid_columnconfigure(1, weight=1)
        self.btn_open_frontend = ctk.CTkButton(actions, text="Open Frontend", command=self._open_frontend)
        self.btn_open_frontend.grid(row=0, column=0, sticky="ew", padx=(0, 10), pady=2)
        self.btn_open_docs = ctk.CTkButton(actions, text="Open API Docs", command=self._open_docs)
        self.btn_open_docs.grid(row=0, column=1, sticky="ew", padx=(0, 0), pady=2)
        # QR tab content (Canvas-based, fills area, keeps perfect square)
        qr_box = ctk.CTkFrame(tab_qr)
        qr_box.grid(row=0, column=0, sticky="nsew", padx=18, pady=18)
        qr_box.grid_rowconfigure(0, weight=1)
        qr_box.grid_columnconfigure(0, weight=1)
        self.qr_canvas = tk.Canvas(qr_box, bg="#0f172a", highlightthickness=0)
        self.qr_canvas.grid(row=0, column=0, sticky="nsew")
        self._qr_url: Optional[str] = None
        self._qr_photo = None
        # Redraw QR on any resize
        self.qr_canvas.bind("<Configure>", lambda e: self._render_qr_to_canvas())

    def _build_logs(self):
        logs_panel = ctk.CTkFrame(self.window)
        logs_panel.grid(row=2, column=0, sticky="nsew", padx=20, pady=(0, 16))
        logs_panel.grid_rowconfigure(1, weight=1)
        logs_panel.grid_columnconfigure(0, weight=1)

        header = ctk.CTkFrame(logs_panel, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", padx=12, pady=(10, 10))
        header.grid_columnconfigure(0, weight=1)

        lbl = ctk.CTkLabel(header, text="Logs", font=ctk.CTkFont(size=14, weight="bold"))
        lbl.grid(row=0, column=0, sticky="w", padx=(2, 8))

        # Filters
        self.log_level_var = tk.StringVar(value="ALL")
        self.process_filter_var = tk.StringVar(value="ALL")
        level_menu = ctk.CTkOptionMenu(header, values=["ALL", "INFO", "WARNING", "ERROR", "DEBUG"], variable=self.log_level_var)
        level_menu.grid(row=0, column=1, padx=(8, 8))
        self.log_search_var = tk.StringVar()
        self.log_search = ctk.CTkEntry(header, placeholder_text="Filter text (regex)", textvariable=self.log_search_var, width=240)
        self.log_search.grid(row=0, column=2, padx=(8, 2))

        # Text box
        self.log_text = ctk.CTkTextbox(logs_panel, wrap="none", height=220)
        self.log_text.grid(row=1, column=0, sticky="nsew", padx=12, pady=(0, 8))
        self.log_text.configure(state="disabled")
        # Log buffer for export
        self._log_buffer: list[str] = []

        # Bind resize to keep QR scaled and remember last tab
        try:
            self.qr_container.bind("<Configure>", lambda e: self._resize_qr_if_present())
            self.tabview._segmented_button.configure(command=self._on_tab_changed)
        except Exception:
            pass
        # Footer bar (status hint)
        try:
            footer = ctk.CTkFrame(self.window, fg_color="transparent")
            footer.grid(row=3, column=0, sticky="ew", padx=20, pady=(0, 16))
            hint = ctk.CTkLabel(footer, text="Right-click a process for actions • Drag the splitter to resize panels", text_color="#9ca3af")
            hint.grid(row=0, column=0, sticky="w", padx=(2, 2), pady=(4, 2))
        except Exception:
            pass

    # ---------- Callbacks ----------
    def _cb_stop(self):
        if self.on_stop_requested:
            try:
                self.on_stop_requested()
            except Exception as e:
                self.logger.error(f"Stop callback failed: {e}")

    def _cb_restart(self):
        if self.on_restart_requested:
            try:
                self.on_restart_requested()
            except Exception as e:
                self.logger.error(f"Restart callback failed: {e}")

    def _on_window_close(self):
        try:
            self._save_prefs_if_changed()
            if self.on_closing:
                self.on_closing()
        finally:
            try:
                self.window.destroy()
            except Exception:
                pass

    # ---------- Public API ----------
    def show(self):
        try:
            self.window.deiconify()
            self.window.lift()
            self.window.focus_force()
        except Exception as e:
            self.logger.error(f"Error showing dashboard: {e}")

    def hide(self):
        try:
            self.window.withdraw()
        except Exception as e:
            self.logger.error(f"Error hiding dashboard: {e}")

    def destroy(self):
        try:
            self._stop_event.set()
            self.window.destroy()
        except Exception as e:
            self.logger.error(f"Error destroying dashboard: {e}")

    def update_progress(self, step: str, progress: float, status: str = "", error: str = ""):
        try:
            self.lbl_step.configure(text=step if not status else f"{step} — {status}")
            self.progress.set_target(max(0.0, min(1.0, float(progress))))
            if error:
                self.lbl_error.configure(text=f"Error: {error}")
                self.progress.set_color("#ef4444")
            else:
                self.lbl_error.configure(text="")
                self.progress.set_color("#3b82f6")
            self.window.update_idletasks()
        except Exception as e:
            self.logger.error(f"update_progress failed: {e}")

    def update_status(self, backend_status: str, frontend_status: str, tunnel_status: str):
        try:
            self._render_chip(self._chip_backend, backend_status)
            self._render_chip(self._chip_frontend, frontend_status)
            # Tunnel special cases for clarity
            ts = (tunnel_status or "").strip()
            if ts.lower() in ("off", "disabled", "not enabled", "turned off"):
                ts = "Disabled"
            self._render_chip(self._chip_tunnel, ts)
        except Exception as e:
            self.logger.error(f"update_status failed: {e}")

    def update_process_status(self, status_dict: Dict[str, Dict[str, Any]]):
        """Update process table with exact metrics. status_dict is from ProcessManager.get_process_status()."""
        try:
            existing_ids = set(self.proc_tree.get_children(""))
            seen_ids = set()
            for name, info in status_dict.items():
                pid = info.get("pid")
                running = info.get("running")
                returncode = info.get("returncode")
                last_error = "; ".join(info.get("recent_errors", [])[-1:]) if info.get("recent_errors") else ""

                # psutil metrics
                cpu = mem = uptime = port = "—"
                if psutil and pid and psutil.pid_exists(pid):
                    try:
                        p = psutil.Process(pid)
                        cpu_val = p.cpu_percent(interval=None)  # non-blocking; repeated calls improve accuracy over time
                        mem_val = p.memory_percent()
                        cpu = f"{cpu_val:.1f}"
                        mem = f"{mem_val:.1f}"
                        create_time = p.create_time()
                        delta = max(0, time.time() - create_time)
                        # exact uptime HH:MM:SS
                        h = int(delta // 3600)
                        m = int((delta % 3600) // 60)
                        s = int(delta % 60)
                        uptime = f"{h:02d}:{m:02d}:{s:02d}"
                        # Find exact listening ports for this PID
                        ports: list[int] = []
                        try:
                            for c in p.connections(kind='inet'):
                                if getattr(c, 'status', None) == psutil.CONN_LISTEN and c.laddr:
                                    try:
                                        ports.append(int(c.laddr.port))
                                    except Exception:
                                        pass
                        except Exception:
                            pass
                        if ports:
                            port = ",".join(str(x) for x in sorted(set(ports)))
                        else:
                            port = self._infer_port_from_name(name)
                    except Exception:
                        pass

                state = "Running" if running else ("Exited" if returncode is not None else "Unknown")

                row_id = f"row::{name}"
                seen_ids.add(row_id)
                values = (name, pid or "—", state, cpu, mem, uptime, port, last_error)
                if row_id in existing_ids:
                    self.proc_tree.item(row_id, values=values)
                else:
                    tags = ('odd',) if len(self.proc_tree.get_children('')) % 2 else ('even',)
                    self.proc_tree.insert("", tk.END, iid=row_id, values=values, tags=tags)

            # remove stale rows
            for iid in existing_ids - seen_ids:
                self.proc_tree.delete(iid)

        except Exception as e:
            self.logger.error(f"update_process_status failed: {e}")

    def append_process_logs(self, status_dict: Dict[str, Dict[str, Any]]):
        try:
            if not status_dict:
                return
            for name, info in status_dict.items():
                outputs = info.get("recent_output", []) or []
                errors = info.get("recent_errors", []) or []
                for line in outputs:
                    self._append_log(name, "INFO", line)
                for line in errors:
                    self._append_log(name, "ERROR", line)
        except Exception as e:
            self.logger.error(f"append_process_logs failed: {e}")

    def _apply_process_filter(self):
        try:
            pattern = self.proc_filter_var.get()
            matcher = None
            if pattern:
                try:
                    matcher = re.compile(pattern)
                except Exception:
                    matcher = None
            for iid in self.proc_tree.get_children(""):
                values = self.proc_tree.item(iid, 'values')
                name = values[0] if values else ""
                visible = True
                if matcher is not None and name:
                    visible = matcher.search(str(name)) is not None
                self.proc_tree.detach(iid) if not visible else self.proc_tree.reattach(iid, '', 'end')
        except Exception:
            pass

    def _sort_treeview(self, column: str):
        try:
            data = []
            for iid in self.proc_tree.get_children(""):
                values = self.proc_tree.item(iid, 'values')
                data.append((iid, values))
            ascending = self._sort_state.get(column, True)
            index = ["name","pid","state","cpu","mem","uptime","port","last_error"].index(column)
            def keyfunc(item):
                val = item[1][index]
                if column in ("cpu","mem"):
                    try:
                        return float(val)
                    except Exception:
                        return -1.0
                if column == "pid":
                    try:
                        return int(val)
                    except Exception:
                        return -1
                if column == "uptime":
                    try:
                        h,m,s = val.split(":")
                        return int(h)*3600 + int(m)*60 + int(s)
                    except Exception:
                        return -1
                return str(val)
            data.sort(key=keyfunc, reverse=not ascending)
            for pos, (iid, _) in enumerate(data):
                self.proc_tree.move(iid, '', pos)
            self._sort_state[column] = not ascending
        except Exception:
            pass

    def _show_proc_menu(self, event):
        try:
            rowid = self.proc_tree.identify_row(event.y)
            if rowid:
                self.proc_tree.selection_set(rowid)
                self._proc_menu.tk_popup(event.x_root, event.y_root)
        finally:
            try:
                self._proc_menu.grab_release()
            except Exception:
                pass

    def _proc_view_only_logs(self):
        try:
            sel = self.proc_tree.selection()
            if not sel:
                return
            values = self.proc_tree.item(sel[0], 'values')
            name = values[0] if values else None
            if name:
                self.log_search_var.set(re.escape(f"[{name}]"))
        except Exception:
            pass

    def _proc_action(self, action: str):
        try:
            # Placeholder for future wiring to launcher controls
            sel = self.proc_tree.selection()
            if not sel:
                return
            values = self.proc_tree.item(sel[0], 'values')
            name = values[0] if values else None
            if not name:
                return
            self._append_log("Launcher", "INFO", f"Requested {action} on {name}")
            if self.on_process_action:
                try:
                    self.on_process_action(name, action)
                except Exception as e:
                    self._append_log("Launcher", "ERROR", f"Process action failed: {e}")
        except Exception:
            pass

    def show_access_info(self, frontend_url: str, backend_url: str):
        try:
            self.lbl_backend_url.configure(text=f"Backend: {backend_url}")
            self.lbl_frontend_url.configure(text=f"Frontend: {frontend_url}")
            self._health_targets["backend"] = backend_url
            self._health_targets["frontend"] = frontend_url
            # Store URL and trigger canvas render
            self._qr_url = frontend_url
            self._render_qr_to_canvas()
        except Exception as e:
            self.logger.error(f"show_access_info failed: {e}")

    # External actions
    def _open_frontend(self):
        try:
            url = self._health_targets.get("frontend")
            if not url:
                return
            import webbrowser
            webbrowser.open(url)
        except Exception as e:
            self.logger.error(f"Open frontend failed: {e}")

    def _open_docs(self):
        try:
            url = self._health_targets.get("backend")
            if not url:
                return
            if not url.endswith("/docs"):
                url = url.rstrip("/") + "/docs"
            import webbrowser
            webbrowser.open(url)
        except Exception as e:
            self.logger.error(f"Open docs failed: {e}")

    def _export_logs(self):
        try:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = Path(f"launcher_logs_{ts}.txt")
            content = "".join(self._log_buffer)
            path.write_text(content, encoding="utf-8")
            self._append_log("Launcher", "INFO", f"Logs exported to {path}")
        except Exception as e:
            self.logger.error(f"Export logs failed: {e}")

    def _save_health_snapshot(self):
        try:
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            snap = {
                "timestamp": ts,
                "targets": self._health_targets,
                "stats": {
                    k: {"status": v["status"], "latency_ms": v["latency_ms"], "last": v["last"].isoformat() if v["last"] else None}
                    for k, v in self._health_stats.items()
                }
            }
            path = Path(f"health_snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            path.write_text(json.dumps(snap, indent=2), encoding="utf-8")
            self._append_log("Launcher", "INFO", f"Health snapshot saved to {path}")
        except Exception as e:
            self.logger.error(f"Save snapshot failed: {e}")

    # ---------- Preferences (column widths, last tab, filters) ----------
    def _load_prefs_and_apply(self):
        try:
            if self._prefs_path.exists():
                self._prefs = json.loads(self._prefs_path.read_text(encoding="utf-8"))
        except Exception:
            self._prefs = {}
        # Apply prefs
        try:
            cols = ["name","pid","state","cpu","mem","uptime","port","last_error"]
            widths = self._prefs.get("proc_column_widths", {})
            for c in cols:
                w = widths.get(c)
                if isinstance(w, int) and w > 40:
                    self.proc_tree.column(c, width=w)
            # Last tab (Overview/QR)
            last_tab = self._prefs.get("right_tab")
            try:
                if isinstance(last_tab, str) and hasattr(self, 'tabview'):
                    self.tabview.set(last_tab)
            except Exception:
                pass
            # Process filter
            pf = self._prefs.get("proc_filter", "")
            if isinstance(pf, str):
                self.proc_filter_var.set(pf)
            # Log filter
            lf = self._prefs.get("log_filter", "")
            if isinstance(lf, str):
                self.log_search_var.set(lf)
        except Exception:
            pass
        # Snapshot baseline
        self._prefs_snapshot = self._capture_prefs()

        # Track column width changes
        try:
            self.proc_tree.bind("<ButtonRelease-1>", lambda e: self._debounced_capture())
        except Exception:
            pass

    def _debounced_capture(self):
        try:
            if hasattr(self, '_debounce_job') and self._debounce_job:
                self.after_cancel(self._debounce_job)
        except Exception:
            pass
        try:
            self._debounce_job = self.window.after(400, lambda: self._store_current_prefs())
        except Exception:
            pass

    def _capture_prefs(self) -> dict:
        cols = ["name","pid","state","cpu","mem","uptime","port","last_error"]
        widths = {}
        for c in cols:
            try:
                widths[c] = int(self.proc_tree.column(c, 'width'))
            except Exception:
                pass
        prefs = {
            "proc_column_widths": widths,
            "proc_filter": self.proc_filter_var.get(),
            "log_filter": self.log_search_var.get()
        }
        try:
            prefs["right_tab"] = self.tabview.get()
        except Exception:
            pass
        return prefs

    def _store_current_prefs(self):
        self._prefs = self._capture_prefs()
        self._save_prefs_if_changed()

    def _save_prefs_if_changed(self):
        try:
            current = self._capture_prefs()
            if json.dumps(current, sort_keys=True) != json.dumps(self._prefs_snapshot, sort_keys=True):
                self._prefs_path.write_text(json.dumps(current, indent=2), encoding="utf-8")
                self._prefs_snapshot = current
        except Exception:
            pass

    # ---------- QR Canvas rendering ----------
    def _render_qr_to_canvas(self):
        try:
            if not (self._qr_url and qrcode and Image is not None and ImageTk is not None):
                # Show placeholder text if nothing to render
                self.qr_canvas.delete("all")
                w = max(0, int(self.qr_canvas.winfo_width()))
                h = max(0, int(self.qr_canvas.winfo_height()))
                self.qr_canvas.create_text(w//2, h//2, text="QR will appear when ready", fill="#9ca3af")
                return
            cw = max(100, int(self.qr_canvas.winfo_width()))
            ch = max(100, int(self.qr_canvas.winfo_height()))
            size = max(120, min(cw, ch) - 32)
            # Generate crisp QR
            qr = qrcode.QRCode(version=None, error_correction=qrcode.constants.ERROR_CORRECT_M, box_size=1, border=2)
            qr.add_data(self._qr_url)
            qr.make(fit=True)
            img = qr.make_image(fill_color="black", back_color="white")
            img = img.resize((size, size), Image.NEAREST)
            # Center on canvas
            self.qr_canvas.delete("all")
            photo = ImageTk.PhotoImage(img)
            self._qr_photo = photo  # hold reference
            x = cw // 2
            y = ch // 2
            # Background plate
            self.qr_canvas.create_rectangle(x - size//2 - 12, y - size//2 - 12, x + size//2 + 12, y + size//2 + 12, fill="#0b1220", outline="")
            self.qr_canvas.create_image(x, y, image=photo)
        except Exception:
            # Fallback to label text
            try:
                self.qr_canvas.delete("all")
                w = max(0, int(self.qr_canvas.winfo_width()))
                h = max(0, int(self.qr_canvas.winfo_height()))
                self.qr_canvas.create_text(w//2, h//2, text="QR unavailable", fill="#9ca3af")
            except Exception:
                pass

    def show_error(self, error: str):
        try:
            self.lbl_error.configure(text=f"Error: {error}")
            self.progress.configure(progress_color="#ef4444")
            self.window.update_idletasks()
        except Exception as e:
            self.logger.error(f"show_error failed: {e}")

    def hide_error(self):
        try:
            self.lbl_error.configure(text="")
            self.progress.configure(progress_color="#3b82f6")
            self.window.update_idletasks()
        except Exception as e:
            self.logger.error(f"hide_error failed: {e}")

    # ---------- Internal helpers ----------
    def _render_chip(self, chip: Dict[str, Any], status_text: str):
        text_lower = (status_text or "").lower()
        # Defaults (neutral)
        dot_color = "#6b7280"
        bg_color = "#0f172a"
        border_color = "#374151"
        value_color = "#e5e7eb"
        name_color = "#9ca3af"

        # Disabled state distinct from stopped/error
        is_disabled = any(k in text_lower for k in ("disabled", "turned off", "off", "not enabled"))
        if any(k in text_lower for k in ("running", "ready")):
            dot_color = "#22c55e"
            bg_color = "#052e1a"
            border_color = "#14532d"
        elif any(k in text_lower for k in ("starting", "connecting", "booting", "initializing")):
            dot_color = "#f59e0b"
            bg_color = "#3a2505"
            border_color = "#92400e"
        elif is_disabled:
            dot_color = "#64748b"  # slate
            bg_color = "#0b1220"
            border_color = "#334155"
            name_color = "#94a3b8"
            value_color = "#cbd5e1"
            # Normalize text for clarity
            status_text = "Disabled"
        elif any(k in text_lower for k in ("error", "failed", "stopped", "crashed")):
            dot_color = "#ef4444"
            bg_color = "#3b0a0a"
            border_color = "#7f1d1d"

        chip["dot"].configure(text_color=dot_color)
        # Maintain name on left, status/value on right
        try:
            chip["frame"].configure(fg_color=bg_color, border_color=border_color)
        except Exception:
            pass
        chip.get("name_label", chip.get("label")).configure(text=chip["name"], text_color=name_color)
        self._update_chip_value(chip, status_text, value_color)

    def _update_chip_value(self, chip: Dict[str, Any], text: str, text_color: Optional[str] = None):
        try:
            chip["full_value_text"] = text
            if text_color:
                chip["value"].configure(text_color=text_color)
            self._reflow_chip(chip)
        except Exception:
            try:
                chip["value"].configure(text=text)
            except Exception:
                pass

    def _reflow_chip(self, chip: Dict[str, Any]):
        try:
            frame = chip["frame"]
            name_lbl = chip["name_label"]
            dot = chip["dot"]
            value_lbl = chip["value"]
            # Ensure geometry info is available
            frame_w = int(frame.winfo_width() or 0)
            if frame_w <= 1:
                return
            left_w = int(dot.winfo_width() or 0) + int(name_lbl.winfo_width() or 0)
            # Internal paddings and gutters
            reserved = left_w + 10 + 8 + 12 + 12
            avail = max(20, frame_w - reserved)
            full_text = chip.get("full_value_text") or value_lbl.cget("text") or ""
            was_trunc = self._ellipsize_to_width(value_lbl, full_text, avail)
            self._set_tooltip(value_lbl, full_text if was_trunc else None)
        except Exception:
            pass

    def _ellipsize_to_width(self, label_widget: Any, full_text: str, max_width_px: int) -> bool:
        try:
            f = tkfont.Font(root=self.window, font=label_widget.cget("font"))
        except Exception:
            # Fallback: set text and hope layout handles it
            label_widget.configure(text=full_text)
            return False
        if f.measure(full_text) <= max_width_px:
            label_widget.configure(text=full_text)
            return False
        ellipsis = "…"
        low, high = 0, len(full_text)
        while low < high:
            mid = (low + high) // 2
            if f.measure(full_text[:mid] + ellipsis) <= max_width_px:
                low = mid + 1
            else:
                high = mid
        truncated = (full_text[: max(0, low - 1)] + ellipsis) if low > 0 else ellipsis
        label_widget.configure(text=truncated)
        return True

    def _set_tooltip(self, widget: Any, text: Optional[str]):
        try:
            if not hasattr(self, "_tooltips"):
                self._tooltips = {}
            tip = self._tooltips.get(widget)
            if text:
                if tip is None:
                    tip = _Tooltip(widget, text)
                    self._tooltips[widget] = tip
                else:
                    tip.text = text
            else:
                if tip is not None:
                    tip.hide()
                self._tooltips.pop(widget, None)
        except Exception:
            pass

    def _infer_port_from_name(self, name: str) -> str:
        # This displays exact known defaults; the actual bound port is reflected by process info if provided by launcher
        mapping = {"Backend": "8000", "Frontend": "5173"}
        return mapping.get(name, "—")

    def _append_log(self, process: str, level: str, text: str):
        try:
            # Filter
            selected_level = self.log_level_var.get()
            if selected_level != "ALL" and level != selected_level:
                return

            ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            color = "#e5e7eb"
            if level == "ERROR":
                color = "#f87171"
            elif level == "WARNING":
                color = "#f59e0b"
            elif level == "DEBUG":
                color = "#9ca3af"
            line = f"[{ts}] [{process}] [{level}] {text}\n"
            search = self.log_search_var.get()
            if search:
                try:
                    import re
                    if not re.search(search, line):
                        return
                except Exception:
                    # invalid regex → show nothing until corrected
                    return
            self.log_text.configure(state="normal")
            self.log_text.insert("end", line)
            self._log_buffer.append(line)
            try:
                # Tag last line with color
                end_index = self.log_text.index('end-1c')
                start_index = f"{float(end_index.split('.')[0]) - 1}.0"
                tag_name = f"lvl_{level.lower()}"
                self.log_text.tag_config(tag_name, foreground=color)
                self.log_text.tag_add(tag_name, start_index, 'end-1c')
            except Exception:
                pass
            self.log_text.see("end")
            self.log_text.configure(state="disabled")
        except Exception as e:
            self.logger.error(f"append_log failed: {e}")

    # ---------- Health checks ----------
    def _start_periodic_health_checks(self):
        if not requests:
            return

        def worker():
            while not self._stop_event.is_set():
                try:
                    self._poll_health_once("backend")
                    self._poll_health_once("frontend")
                except Exception:
                    pass
                # 2s cadence
                self._stop_event.wait(2.0)

        t = threading.Thread(target=worker, daemon=True)
        t.start()

    def _poll_health_once(self, key: str):
        url = self._health_targets.get(key)
        if not url:
            return
        # Endpoint selection
        target = url if key == "frontend" else f"{url.rstrip('/')}/docs"
        start = time.perf_counter()
        status = "Unknown"
        latency_ms: Optional[float] = None
        try:
            resp = requests.get(target, timeout=3)
            latency_ms = (time.perf_counter() - start) * 1000.0
            status = f"HTTP {resp.status_code}"
        except requests.Timeout:
            latency_ms = (time.perf_counter() - start) * 1000.0
            status = "Timeout"
        except Exception as e:
            latency_ms = (time.perf_counter() - start) * 1000.0
            status = f"Error"

        self._health_stats[key] = {
            "status": status,
            "latency_ms": latency_ms,
            "last": datetime.now(),
        }

        # Update chips and labels on UI thread
        def render():
            stats = self._health_stats[key]
            last_str = stats["last"].strftime("%H:%M:%S") if stats["last"] else "—"
            lat_str = f"{stats['latency_ms']:.1f} ms" if stats["latency_ms"] is not None else "—"
            if key == "backend":
                self._update_chip_value(self._chip_backend_latency, lat_str)
                self.lbl_backend_health.configure(text=f"Status: {stats['status']} | Latency: {lat_str} | Last: {last_str}")
                self._push_latency_point('backend', stats['latency_ms'])
            else:
                self._update_chip_value(self._chip_frontend_latency, lat_str)
                self.lbl_frontend_health.configure(text=f"Status: {stats['status']} | Latency: {lat_str} | Last: {last_str}")
                self._push_latency_point('frontend', stats['latency_ms'])

        try:
            self.window.after(0, render)
        except Exception:
            pass

    # ---------- Progress and sparkline animations ----------
    def _start_progress_animation(self):
        def tick():
            try:
                if hasattr(self, 'progress'):
                    self.progress.tick()
            finally:
                self.window.after(16, tick)  # ~60fps
        self.window.after(16, tick)

    def _push_latency_point(self, key: str, ms: Optional[float]):
        if ms is None:
            return
        hist = self._latency_history.get(key)
        if hist is None:
            return
        hist.append(ms)
        if len(hist) > 60:
            del hist[0]
        self._redraw_sparkline(key)

    def _redraw_sparkline(self, key: str):
        canvas = self.backend_spark if key == 'backend' else self.frontend_spark
        hist = self._latency_history[key]
        canvas.delete("all")
        if not hist:
            return
        w = int(canvas.winfo_width() or 1)
        h = int(canvas.winfo_height() or 1)
        n = len(hist)
        max_v = max(1.0, max(hist))
        pts = []
        for i, v in enumerate(hist):
            x = int(i * (w - 6) / max(1, n - 1)) + 3
            y = int(h - 6 - (v / max_v) * (h - 12)) + 3
            pts.append((x, y))
        # gradient-like stroke using two passes
        if len(pts) >= 2:
            for off, col, wdt in ((1, '#0ea5e9', 3), (0, '#38bdf8', 1)):
                shifted = [(x, y + off) for (x, y) in pts]
                canvas.create_line(*sum(shifted, ()), fill=col, width=wdt, smooth=True)


class _SmoothProgress(ctk.CTkFrame):
    """Custom smooth progress bar with eased interpolation and accent stripe."""

    def __init__(self, parent, height: int = 16, color: str = "#3b82f6"):
        super().__init__(parent, fg_color="#0b1220")
        self._canvas = tk.Canvas(self, height=height, bg="#0b1220", highlightthickness=0)
        self._canvas.pack(fill="x", expand=True)
        self._target = 0.0
        self._value = 0.0
        self._color = color

    def set_target(self, val: float):
        self._target = max(0.0, min(1.0, float(val)))

    def set_color(self, color: str):
        self._color = color

    def tick(self):
        # ease towards target
        delta = self._target - self._value
        if abs(delta) > 0.0005:
            self._value += delta * 0.15  # easing
        else:
            self._value = self._target
        self._redraw()

    def grid(self, *args, **kwargs):  # ensure layout works like a widget
        return super().grid(*args, **kwargs)

    def _redraw(self):
        c = self._canvas
        c.delete("all")
        w = int(c.winfo_width() or 1)
        h = int(c.winfo_height() or 1)
        # track
        c.create_rectangle(0, h//2 - (h//2 - 2), w, h//2 + (h//2 - 2), fill="#0f172a", outline="")
        # bar
        filled = int(w * self._value)
        if filled > 0:
            c.create_rectangle(2, 2, filled-2, h-2, fill=self._color, outline="")
            # accent moving stripe
            stripe_w = max(12, w // 12)
            phase = int((time.time() * 120) % (stripe_w * 2))
            for x in range(2 - phase, filled - 2, stripe_w):
                c.create_rectangle(x, 2, min(x + stripe_w//3, filled-2), h-2, fill="#60a5fa", outline="")




class _Tooltip:
    """Simple tooltip that shows on hover. Hidden when no text or on leave."""
    def __init__(self, widget: Any, text: str):
        self.widget = widget
        self.text = text
        self._tipwin = None
        try:
            widget.bind("<Enter>", self._show)
            widget.bind("<Leave>", self._hide)
        except Exception:
            pass

    def _show(self, event=None):
        if not self.text:
            return
        try:
            if self._tipwin is not None:
                self.hide()
            x = self.widget.winfo_rootx() + 10
            y = self.widget.winfo_rooty() + self.widget.winfo_height() + 4
            self._tipwin = tw = tk.Toplevel(self.widget)
            tw.wm_overrideredirect(True)
            tw.wm_geometry(f"+{x}+{y}")
            lbl = tk.Label(tw, text=self.text, justify=tk.LEFT,
                           background="#111827", foreground="#e5e7eb",
                           relief=tk.SOLID, borderwidth=1,
                           font=("TkDefaultFont", 9))
            lbl.pack(ipadx=6, ipady=4)
        except Exception:
            self._tipwin = None

    def _hide(self, event=None):
        self.hide()

    def hide(self):
        try:
            if self._tipwin is not None:
                self._tipwin.destroy()
        finally:
            self._tipwin = None
