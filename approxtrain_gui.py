#ApproxTrain GUI, built with Tkinter. Allows building simple Keras models with approximate layers, running training scripts, and visualizing metrics.
#Note: this is a demo GUI for ApproxTrain, not a general-purpose Keras model builder. 
# Generates simple scripts with a fixed training loop and limited layer types, designed to be compatible with ApproxTrain's approximate layers and multiplier LUTs. 
# The "Train" page allows running arbitrary scripts but without the code generation features.

#Developed by Sasha Odumodu and Khadijah Bashir
#AI Disclosure: Traditional as well as AI-generated code was used in the development of this GUI.
#All credits go to the original ApproxTrain team for their work on the underlying training framework and approximate layers.

#For future contributors, it is encouraged to add more to the options frame for accessibility, and expand model maker's capabilities.

import os
import sys
import re
import math
import time
import queue
import threading
import subprocess
import tempfile
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinter import font as tkfont
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

REPO_ROOT = Path(__file__).resolve().parent

LUT_FILES = {p.name: p for p in sorted((REPO_ROOT / "lut").glob("*.bin"))}

#Layer definitions for the Model maker page, following basic keras layer parameters.
LAYER_DEFS = {
    "Dense": [
        {"name": "units",      "label": "Units",      "type": "int",    "default": 128},
        {"name": "activation", "label": "Activation", "type": "choice", "default": "relu",
         "choices": ["relu", "sigmoid", "softmax", "tanh", "linear"]},
    ],
    "Flatten": [],
    "Dropout": [
        {"name": "rate", "label": "Rate", "type": "float", "default": 0.5},
    ],
    "Conv2D": [
        {"name": "filters",     "label": "Filters",    "type": "int",    "default": 32},
        {"name": "kernel_size", "label": "Kernel",     "type": "int",    "default": 3},
        {"name": "activation",  "label": "Activation", "type": "choice", "default": "relu",
         "choices": ["relu", "sigmoid", "tanh", "linear"]},
    ],
    "MaxPooling2D": [
        {"name": "pool_size", "label": "Pool Size", "type": "int", "default": 2},
    ],
    "BatchNormalization": [],
    "Input (Flatten 28×28)": [],
}
LAYER_NAMES = list(LAYER_DEFS.keys())


# ─────────────────────────────────────────────────────────────────────────────
# Metrics collection
# Parses stdout lines from Keras during training
# ─────────────────────────────────────────────────────────────────────────────

#Metric keys are what are looked for in the keras output during model training. The GUI looks for these keys to populate the plot and csv export data
METRIC_KEYS = ["epoch", "elapsed_s", "loss", "accuracy", "val_loss", "val_accuracy"]


class MetricsCollector:
    """Parses Keras stdout lines and accumulates per-epoch metrics."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.data  = {k: [] for k in METRIC_KEYS}
        self._epoch = 0 #current epoch being parsed
        self._t0    = None #timestamp of first epoch, for elapsed time calculation
        self.dirty  = False # whether new data has arrived since last plot refresh

    def feed(self, line: str):
        # "Epoch 3/10"
        m = re.match(r'Epoch (\d+)/\d+', line.strip()) #look for epoch number in the line
        if m:
            self._epoch = int(m.group(1))
            if self._t0 is None:
                self._t0 = time.time()
            return

        # Final step summary: contains "/step" but not "ETA"
        if self._epoch > 0 and '/step' in line and 'ETA' not in line:
            # Map verbose Keras metric names to canonical keys, since different scripts may use different metrics or formats.
            aliases = {
                'loss':                          'loss',
                'sparse_categorical_crossentropy': 'loss',
                'binary_crossentropy':            'loss',
                'accuracy':                       'accuracy',
                'sparse_categorical_accuracy':    'accuracy',
                'categorical_accuracy':           'accuracy',
                'val_loss':                       'val_loss',
                'val_sparse_categorical_crossentropy': 'val_loss',
                'val_binary_crossentropy':        'val_loss',
                'val_accuracy':                   'val_accuracy',
                'val_sparse_categorical_accuracy': 'val_accuracy',
                'val_categorical_accuracy':       'val_accuracy',
            }
            vals = {}
            for raw, canonical in aliases.items():
                if canonical in vals:
                    continue
                m2 = re.search(rf'\b{re.escape(raw)}: ([0-9.eE+\-]+)', line) 
                if m2:
                    try:
                        vals[canonical] = float(m2.group(1))
                    except ValueError:
                        pass
            if vals:
                elapsed = round(time.time() - self._t0, 2) if self._t0 else 0.0
                self.data["epoch"].append(self._epoch)
                self.data["elapsed_s"].append(elapsed)
                for key in ('loss', 'accuracy', 'val_loss', 'val_accuracy'):
                    self.data[key].append(vals.get(key, float('nan')))
                self.dirty  = True
                self._epoch = 0  # prevent evaluate/extra /step lines being recorded


# ─────────────────────────────────────────────────────────────────────────────
# Plot panel
# This is the embedded matplotlib graphing panel used for visualizing training metrics.
# Allows the user to select metrics to plot on X, Y, and Y2
# Refreshes automatically when new data arrives
# ─────────────────────────────────────────────────────────────────────────────

_Y2_NONE = "— none —"


class PlotPanel(tk.Frame):
    """Embedded matplotlib figure with user-selectable X / Y / Y2 axes."""

    def __init__(self, parent, metrics: MetricsCollector):
        super().__init__(parent)
        self._metrics = metrics
        self._build()

    def _build(self):
        ctrl = tk.Frame(self)
        ctrl.pack(fill="x", padx=6, pady=4)

        for label, attr, default, extra in [
            ("X",  "_x_var",  "epoch",    []),
            ("Y",  "_y_var",  "loss",     []),
            ("Y2", "_y2_var", _Y2_NONE,   [_Y2_NONE]),
        ]:
            tk.Label(ctrl, text=label).pack(side="left", padx=(8, 2))
            var = tk.StringVar(value=default)
            setattr(self, attr, var)
            ttk.Combobox(ctrl, textvariable=var,
                         values=extra + METRIC_KEYS,
                         state="readonly", width=13).pack(side="left")

        ttk.Button(ctrl, text="Refresh", command=self.refresh).pack(side="left", padx=10)

        fig = Figure(figsize=(5, 3.2), dpi=90, tight_layout=True)
        self._ax     = fig.add_subplot(111)
        self._canvas = FigureCanvasTkAgg(fig, master=self)
        self._canvas.get_tk_widget().pack(fill="both", expand=True)

        self.refresh()

    def refresh(self):
        d = self._metrics.data
        x_key  = self._x_var.get()
        y_key  = self._y_var.get()
        y2_key = self._y2_var.get()

        def pairs(xk, yk):
            return [
                (x, y) for x, y in zip(d.get(xk, []), d.get(yk, []))
                if not (isinstance(x, float) and math.isnan(x))
                and not (isinstance(y, float) and math.isnan(y))
            ]

        self._ax.clear()
        plotted = False

        p1 = pairs(x_key, y_key)
        if p1:
            xs, ys = zip(*p1)
            self._ax.plot(xs, ys, marker='o', linewidth=1.5, label=y_key)
            plotted = True

        if y2_key != _Y2_NONE:
            p2 = pairs(x_key, y2_key)
            if p2:
                xs2, ys2 = zip(*p2)
                self._ax.plot(xs2, ys2, marker='s', linewidth=1.5,
                              linestyle='--', label=y2_key)
                plotted = True

        if plotted:
            self._ax.set_xlabel(x_key)
            self._ax.set_ylabel("value" if y2_key != _Y2_NONE else y_key)
            self._ax.grid(True, alpha=0.3)
            self._ax.legend(fontsize=8)
        else:
            self._ax.text(0.5, 0.5,
                          "No data yet.\nRun training to populate the plot.",
                          transform=self._ax.transAxes,
                          ha='center', va='center', color='gray', fontsize=10)

        self._canvas.draw()


# ─────────────────────────────────────────────────────────────────────────────
# Shared runner mixin
# Provides subprocess management by feeding stdout lines into a queue, which are then processed
# Threads are important in this step to avoid blocking the GUI while waiting for the training to fully complete.
# ─────────────────────────────────────────────────────────────────────────────

class RunnerMixin:
    """Subprocess management + metrics feeding + auto plot refresh."""

    #Called by frames to set up subprocess management.
    def _init_runner(self, log_widget: tk.Text, metrics: MetricsCollector):
        self._log            = log_widget #where output lines are written
        self._queue          = queue.Queue()
        self._process        = None
        self._metrics        = metrics
        self._plot_panel     = None   # set by frame after PlotPanel is created
        self._last_plot_time = 0.0
        self._poll()

    #Called in the main thread to process lines from subprocess and update
    def _poll(self):
        while not self._queue.empty():
            line = self._queue.get()
            self._write(line)
            self._metrics.feed(line)

        # Refresh plot at most every 2 s when new data is available
        now = time.time()
        if (self._plot_panel is not None
                and self._metrics.dirty
                and now - self._last_plot_time > 2.0):
            self._plot_panel.refresh()
            self._metrics.dirty  = False
            self._last_plot_time = now

        self.after(100, self._poll)

    #Write text to widget
    def _write(self, text: str):
        self._log.insert(tk.END, text)
        self._log.see(tk.END)

    #Run the command in  a subprocess and feed output to queue
    def _run(self, cmd):
        if self._process is not None and self._process.poll() is None:
            messagebox.showwarning("Already running", "A process is already running.")
            return
        self._metrics.reset()
        self._write(f"\n$ {' '.join(cmd)}\n")
        threading.Thread(target=self._run_thread, args=(cmd,), daemon=True).start()

    #Target of thread that runs the subprocess
    def _run_thread(self, cmd):
        try:
            self._process = subprocess.Popen(
                cmd, cwd=REPO_ROOT,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                universal_newlines=True, bufsize=1,
            )
            assert self._process.stdout is not None
            for line in self._process.stdout:
                self._queue.put(line)
            code = self._process.wait()
            self._queue.put(f"\n[Process exited with code {code}]\n")
        except Exception as e:
            self._queue.put(f"\n[Error: {e}]\n")
        finally:
            self._process = None

    #If "stop" is pressed, it terminates the subprocess, as long as it's running.
    #Writes a log, "Stop requested" as user feedback
    def _stop(self):
        if self._process is not None and self._process.poll() is None:
            self._process.terminate()
            self._write("\n[Stop requested]\n")

    #Used to save the CSV from the collected metrics from the model output.
    def _save_csv(self):
        import csv
        data = self._metrics.data
        epochs = data.get("epoch", [])
        if not epochs:
            messagebox.showinfo("No data", "No metrics collected yet.")
            return
        # Ask the user where to save the file
        path = filedialog.asksaveasfilename(
            title="Save metrics CSV", defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")], initialdir=REPO_ROOT)
        if not path:
            return
        keys = [k for k in METRIC_KEYS if k != "epoch"]
        # Write the CSV with header and rows per epoch. If there are missing values, fill as an empty string
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch"] + keys)
            for i, epoch in enumerate(epochs):
                writer.writerow([epoch] + [data[k][i] if i < len(data[k]) else "" for k in keys])
        #Confirm file saved
        messagebox.showinfo("Saved", f"Metrics saved to:\n{path}")

    def _make_output_notebook(self, parent) -> tk.Text:
        """
        Build a ttk.Notebook with Log and Plot tabs.
        Returns the log Text widget. Sets self._plot_panel as a side-effect.
        Call this after self._metrics exists.
        """
        nb = ttk.Notebook(parent)
        nb.pack(fill="both", expand=True)

        # Log tab
        log_frame = tk.Frame(nb)
        nb.add(log_frame, text="Log")
        log = tk.Text(log_frame, wrap="word", font=self.app.fonts["mono"])
        log.pack(side="left", fill="both", expand=True)
        sb = ttk.Scrollbar(log_frame, command=log.yview)
        sb.pack(side="right", fill="y")
        log.configure(yscrollcommand=sb.set)

        # Plot tab
        plot_frame = tk.Frame(nb)
        nb.add(plot_frame, text="Plot")
        panel = PlotPanel(plot_frame, self._metrics)
        panel.pack(fill="both", expand=True)
        self._plot_panel = panel

        return log


# ─────────────────────────────────────────────────────────────────────────────
# App controller
# This is the main application class that initalizes the window and acts as a manager for different pages/frames.
# Handles font size, for accessibility. 
# Cleans up any temporary model maker files on startup.
# ─────────────────────────────────────────────────────────────────────────────

class ApproxTrainGUI:
    BASE_FONT_SIZE = 12 #Default font size

    def __init__(self, root):
        self.root = root
        self.root.title("ApproxTrain")
        self.root.geometry("960x680")
        self._init_fonts(self.BASE_FONT_SIZE)
        #Cleaning temporary model maker files
        for f in REPO_ROOT.glob("_approxtrain_maker_*.py"):
            try:
                f.unlink()
            except FileNotFoundError:
                pass

        self.container = tk.Frame(self.root)
        self.container.pack(fill="both", expand=True)
        self.container.grid_rowconfigure(0, weight=1)
        self.container.grid_columnconfigure(0, weight=1)

        #Frame initialization
        self.frames = {}
        for cls in (MainMenuFrame, TrainFrame,
                    ModelMakerFrame, CreditsFrame, OptionsFrame):
            f = cls(self.container, self)
            self.frames[cls] = f
            f.grid(row=0, column=0, sticky="nsew")

        self.show_frame(MainMenuFrame)

    def show_frame(self, cls):
        self.frames[cls].tkraise()

    #Font management
    def _init_fonts(self, base):
        self.fonts = {
            "title":    tkfont.Font(family="Helvetica", size=base * 4,     weight="bold"),
            "page":     tkfont.Font(family="Helvetica", size=base * 2,     weight="bold"),
            "section":  tkfont.Font(family="Helvetica", size=base + 4,     weight="bold"),
            "subhead":  tkfont.Font(family="Helvetica", size=base - 1,     weight="bold"),
            "body":     tkfont.Font(family="Helvetica", size=base),
            "mono":     tkfont.Font(family="Courier",   size=max(base - 3, 7)),
        }

    #Manages the changing of the font size of the GUI
    def set_base_font_size(self, base):
        self.fonts["title"].configure(size=base * 4)
        self.fonts["page"].configure(size=base * 2)
        self.fonts["section"].configure(size=base + 4)
        self.fonts["subhead"].configure(size=max(base - 1, 7))
        self.fonts["body"].configure(size=base)
        self.fonts["mono"].configure(size=max(base - 3, 7))


# ─────────────────────────────────────────────────────────────────────────────
# Main menu
# Simple main menu, first thing on start up.
# ─────────────────────────────────────────────────────────────────────────────

class MainMenuFrame(tk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        self._build()

    def _build(self):
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        tk.Label(self, text="ApproxTrain",
                 font=self.app.fonts["title"]).grid(row=0, column=0, pady=(80, 20))

        btn_frame = tk.Frame(self)
        btn_frame.grid(row=1, column=0)

        for label, cls in [
            ("Credits",     CreditsFrame),
            ("Train",       TrainFrame),
            ("Model Maker", ModelMakerFrame),
            ("Options",     OptionsFrame),
        ]:
            tk.Button(btn_frame, text=label, width=12, height=2,
                      command=lambda c=cls: self.app.show_frame(c)
                      ).pack(side="left", padx=12)


# ─────────────────────────────────────────────────────────────────────────────
# Train
# Allows the user to select a training script, a look up table, and the option to enable/disable approximate mode.
# ─────────────────────────────────────────────────────────────────────────────

class TrainFrame(RunnerMixin, tk.Frame):
    def __init__(self, parent, app):
        tk.Frame.__init__(self, parent)
        self.app = app
        lut_default = "lut/MBM_7.bin" if (REPO_ROOT / "lut" / "MBM_7.bin").exists() else ""
        self._script_var  = tk.StringVar(value="lenet300100.py")
        self._lut_var     = tk.StringVar(value=lut_default)
        self._approx_var  = tk.BooleanVar(value=True)
        self._preview_var = tk.StringVar()
        self._metrics     = MetricsCollector()
        self._build()
        self._refresh_preview()

    #Builds the UI components for the Train page
    def _build(self):
        main = ttk.Frame(self, padding=12)
        main.pack(fill="both", expand=True)

        ttk.Button(main, text="← Menu",
                   command=lambda: self.app.show_frame(MainMenuFrame)).pack(anchor="w")

        cfg = ttk.LabelFrame(main, text="Train Configuration", padding=10)
        cfg.pack(fill="x", pady=(6, 0))
        cfg.columnconfigure(1, weight=1)

        ttk.Label(cfg, text="Script").grid(row=0, column=0, sticky="w", padx=5, pady=6)
        ttk.Entry(cfg, textvariable=self._script_var
                  ).grid(row=0, column=1, sticky="ew", padx=5, pady=6)
        ttk.Button(cfg, text="Browse", command=self._browse_script
                   ).grid(row=0, column=2, padx=5, pady=6)

        ttk.Label(cfg, text="Multiplier LUT").grid(row=1, column=0, sticky="w", padx=5, pady=6)
        ttk.Entry(cfg, textvariable=self._lut_var
                  ).grid(row=1, column=1, sticky="ew", padx=5, pady=6)
        ttk.Button(cfg, text="Browse", command=self._browse_lut
                   ).grid(row=1, column=2, padx=5, pady=6)

        ttk.Checkbutton(cfg, text="Approximate mode",
                        variable=self._approx_var
                        ).grid(row=2, column=1, sticky="w", padx=5, pady=6)

        for v in (self._script_var, self._lut_var, self._approx_var):
            v.trace_add("write", lambda *_: self._refresh_preview())

        prev = ttk.LabelFrame(main, text="Command Preview", padding=6)
        prev.pack(fill="x", pady=(8, 0))
        ttk.Label(prev, textvariable=self._preview_var,
                  font=self.app.fonts["mono"], anchor="w", wraplength=900).pack(fill="x")

        btns = ttk.Frame(main)
        btns.pack(fill="x", pady=(8, 4))
        ttk.Button(btns, text="Run",      command=self._do_run).pack(side="left", padx=5)
        ttk.Button(btns, text="Stop",     command=self._stop).pack(side="left", padx=5)
        ttk.Button(btns, text="Save CSV", command=self._save_csv).pack(side="left", padx=5)
        ttk.Button(btns, text="Clear Log",
                   command=lambda: self._log.delete("1.0", tk.END)).pack(side="left", padx=5)

        log = self._make_output_notebook(main)
        self._init_runner(log, self._metrics)

    #File browsing for the training script
    def _browse_script(self):
        p = filedialog.askopenfilename(
            title="Select Python script",
            initialdir=REPO_ROOT,
            filetypes=[("Python files", "*.py"), ("All files", "*.*")])
        if p:
            try:
                self._script_var.set(str(Path(p).relative_to(REPO_ROOT)))
            except ValueError:
                self._script_var.set(p)

    #File browsing for the LUT
    def _browse_lut(self):
        p = filedialog.askopenfilename(
            title="Select LUT file",
            initialdir=REPO_ROOT / "lut",
            filetypes=[("Binary files", "*.bin"), ("All files", "*.*")])
        if p:
            try:
                self._lut_var.set(str(Path(p).relative_to(REPO_ROOT)))
            except ValueError:
                self._lut_var.set(p)

    #Builds the command to run the training script based on selection
    def _build_cmd(self):
        script = self._script_var.get().strip()
        if not script:
            raise ValueError("No script selected.")
        cmd = [sys.executable, "-u",script]
        lut = self._lut_var.get().strip()
        if lut:
            cmd += ["--mul", lut]
        if self._approx_var.get():
            cmd += ["--approx"]
        return cmd

    #Refresh command preview based on the user's current selection
    def _refresh_preview(self):
        try:
            self._preview_var.set(" ".join(self._build_cmd()))
        except Exception as e:
            self._preview_var.set(f"(error: {e})")

    #Try to run the script, unless there is an error. Also checks to be sure that the script exists.
    def _do_run(self):
        try:
            cmd = self._build_cmd()
        except Exception as e:
            messagebox.showerror("Error", str(e))
            return
        if not (REPO_ROOT / cmd[2]).exists():
            messagebox.showerror("Error", f"Script not found:\n{cmd[2]}")
            return
        self._run(cmd)


# ─────────────────────────────────────────────────────────────────────────────
# Model Maker — layer rows
# UI components for the Model maker for allowing the user to build their own model
# ─────────────────────────────────────────────────────────────────────────────

class LayerRow(tk.Frame):
    def __init__(self, parent, on_change, on_remove, initial_type=None):
        super().__init__(parent, relief="groove", bd=1, padx=4, pady=3)
        self._on_change  = on_change
        self._on_remove  = on_remove
        self._param_vars = {}

        self._type_var = tk.StringVar(value=initial_type or LAYER_NAMES[0])
        self._type_var.trace_add("write", lambda *_: self._rebuild_params())

        ttk.Combobox(self, textvariable=self._type_var,
                     values=LAYER_NAMES, state="readonly", width=20).pack(side="left", padx=(0, 6))

        self._param_frame = tk.Frame(self)
        self._param_frame.pack(side="left", fill="x", expand=True)

        tk.Button(self, text="×", fg="red", width=2,
                  relief="flat", command=self._on_remove).pack(side="right", padx=(6, 0))

        self._rebuild_params()

    def _rebuild_params(self):
        for w in self._param_frame.winfo_children():
            w.destroy()
        self._param_vars.clear()

        for p in LAYER_DEFS.get(self._type_var.get(), []):
            tk.Label(self._param_frame, text=p["label"]).pack(side="left", padx=(6, 2))
            var = tk.StringVar(value=str(p["default"]))
            self._param_vars[p["name"]] = var
            if p["type"] == "choice":
                w = ttk.Combobox(self._param_frame, textvariable=var,
                                 values=p["choices"], state="readonly", width=10)
            else:
                w = ttk.Entry(self._param_frame, textvariable=var, width=7)
            w.pack(side="left", padx=2)
            var.trace_add("write", lambda *_: self._on_change())

        self._on_change()

    def get_config(self):
        return {"type": self._type_var.get(),
                "params": {k: v.get() for k, v in self._param_vars.items()}}


# ─────────────────────────────────────────────────────────────────────────────
# Model Maker frame
# Handles the brunt of the logic, such as the script generation.
# ─────────────────────────────────────────────────────────────────────────────

#Basic model maker datasets. The user is also allowed to select "Custom" to pull from another dataset, but these are defaults for ease of use.
MAKER_DATASETS = {
    "MNIST":         "mnist",
    "Fashion-MNIST": "fashion_mnist",
    "KMNIST":        "kmnist",
}

#Main frame
class ModelMakerFrame(RunnerMixin, tk.Frame):
    def __init__(self, parent, app):
        tk.Frame.__init__(self, parent)
        self.app          = app
        self._rows               = []
        self._dataset_var        = tk.StringVar(value="MNIST")
        self._custom_dataset_var = tk.StringVar()
        self._epochs_var         = tk.IntVar(value=5)
        self._batch_var          = tk.IntVar(value=128)
        self._lr_var             = tk.StringVar(value="0.001")
        lut_default = "lut/MBM_7.bin" if (REPO_ROOT / "lut" / "MBM_7.bin").exists() else ""
        self._approx_var         = tk.BooleanVar(value=False)
        self._lut_var            = tk.StringVar(value=lut_default)
        self._metrics            = MetricsCollector()
        self._build()
        for t in ("Input (Flatten 28×28)", "Dense", "Dense"):
            self._add_row(initial_type=t)
        self._add_output_row()

    #Builds UI components
    def _build(self):
        top = ttk.Frame(self, padding=(12, 8, 12, 0))
        top.pack(fill="x")
        ttk.Button(top, text="← Menu",
                   command=lambda: self.app.show_frame(MainMenuFrame)).pack(side="left")
        tk.Label(top, text="Model Maker",
                 font=self.app.fonts["section"]).pack(side="left", padx=16)

        params = ttk.Frame(self, padding=(12, 0, 12, 4))
        params.pack(fill="x")

        tk.Label(params, text="Dataset:").pack(side="left", padx=(0, 4))
        ds_frame = ttk.Frame(params)
        ds_frame.pack(side="left")
        ttk.Combobox(ds_frame, textvariable=self._dataset_var,
                     values=list(MAKER_DATASETS.keys()) + ["Custom…"],
                     state="readonly", width=14).pack(side="left")
        self._custom_entry = ttk.Entry(ds_frame, textvariable=self._custom_dataset_var, width=16)
        self._custom_entry.pack(side="left", padx=(4, 0))
        self._custom_entry.pack_forget()
        self._dataset_var.trace_add("write", lambda *_: self._on_dataset_change())
        self._custom_dataset_var.trace_add("write", lambda *_: self._refresh_code())

        tk.Label(params, text="Epochs:").pack(side="left", padx=(16, 4))
        tk.Spinbox(params, textvariable=self._epochs_var, from_=1, to=1000,
                   width=5, command=self._refresh_code).pack(side="left")
        self._epochs_var.trace_add("write", lambda *_: self._refresh_code())

        tk.Label(params, text="Batch:").pack(side="left", padx=(12, 4))
        ttk.Combobox(params, textvariable=self._batch_var,
                     values=[32, 64, 128, 256, 512], width=6).pack(side="left")
        self._batch_var.trace_add("write", lambda *_: self._refresh_code())

        tk.Label(params, text="LR:").pack(side="left", padx=(12, 4))
        ttk.Combobox(params, textvariable=self._lr_var,
                     values=["0.1", "0.01", "0.001", "0.0001", "0.00001"], width=8).pack(side="left")
        self._lr_var.trace_add("write", lambda *_: self._refresh_code())

        approx_row = ttk.Frame(self, padding=(12, 0, 12, 6))
        approx_row.pack(fill="x")
        ttk.Checkbutton(approx_row, text="Approximate mode",
                        variable=self._approx_var,
                        command=self._refresh_code).pack(side="left")
        tk.Label(approx_row, text="LUT:").pack(side="left", padx=(16, 4))
        ttk.Entry(approx_row, textvariable=self._lut_var, width=40).pack(side="left")
        ttk.Button(approx_row, text="Browse", command=self._browse_lut).pack(side="left", padx=(4, 0))
        self._lut_var.trace_add("write", lambda *_: self._refresh_code())

        paned = tk.PanedWindow(self, orient="horizontal",
                               sashrelief="raised", sashwidth=5)
        paned.pack(fill="both", expand=True, padx=12, pady=8)

        # ── Left: layer stack ─────────────────────────────────────────────
        left = ttk.Frame(paned)
        paned.add(left, minsize=360)

        ttk.Label(left, text="Layers",
                  font=self.app.fonts["subhead"]).pack(anchor="w", pady=(0, 4))

        canvas_outer = tk.Frame(left, relief="sunken", bd=1)
        canvas_outer.pack(fill="both", expand=True)

        self._canvas = tk.Canvas(canvas_outer, highlightthickness=0)
        vsb = ttk.Scrollbar(canvas_outer, orient="vertical", command=self._canvas.yview)
        self._canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        self._canvas.pack(side="left", fill="both", expand=True)

        self._stack = tk.Frame(self._canvas)
        self._cwin  = self._canvas.create_window((0, 0), window=self._stack, anchor="nw")
        self._stack.bind("<Configure>",
                         lambda _: self._canvas.configure(
                             scrollregion=self._canvas.bbox("all")))
        self._canvas.bind("<Configure>",
                          lambda e: self._canvas.itemconfig(self._cwin, width=e.width))

        ttk.Button(left, text="+ Add Layer", command=self._add_row).pack(pady=6)

        # ── Right: code preview + log/plot notebook ───────────────────────
        right = ttk.Frame(paned)
        paned.add(right, minsize=340)

        ttk.Label(right, text="Generated Script",
                  font=self.app.fonts["subhead"]).pack(anchor="w", pady=(0, 4))

        self._code_box = tk.Text(right, wrap="none", font=self.app.fonts["mono"],
                                 state="disabled", height=14)
        self._code_box.pack(fill="x")
        xsb = ttk.Scrollbar(right, orient="horizontal", command=self._code_box.xview)
        xsb.pack(fill="x")
        self._code_box.configure(xscrollcommand=xsb.set)

        btns = ttk.Frame(right)
        btns.pack(fill="x", pady=(6, 4))
        ttk.Button(btns, text="Run",         command=self._do_run).pack(side="left", padx=5)
        ttk.Button(btns, text="Stop",        command=self._stop).pack(side="left", padx=5)
        ttk.Button(btns, text="Save Script", command=self._save).pack(side="left", padx=5)
        ttk.Button(btns, text="Save CSV",    command=self._save_csv).pack(side="left", padx=5)
        ttk.Button(btns, text="Clear Log",
                   command=lambda: self._log.delete("1.0", tk.END)).pack(side="left", padx=5)

        log = self._make_output_notebook(right)
        self._init_runner(log, self._metrics)

    # ── Layer management ──────────────────────────────────────────────────────

    #For adding a new layer to the model architecture
    def _add_output_row(self):
        row = LayerRow(self._stack,
                       on_change=self._refresh_code,
                       on_remove=lambda r=None: self._remove_row(row),
                       initial_type="Dense")
        row.pack(fill="x", pady=2, padx=2)
        self._rows.append(row)
        try:
            row._param_vars["units"].set(10)
            row._param_vars["activation"].set("softmax")
        except (KeyError, tk.TclError):
            pass
        self._refresh_code()

    #For selecting the LUT for approximate mode, with file browsing
    def _browse_lut(self):
        p = filedialog.askopenfilename(
            title="Select LUT file", initialdir=REPO_ROOT / "lut",
            filetypes=[("Binary files", "*.bin"), ("All files", "*.*")])
        if p:
            try:
                self._lut_var.set(str(Path(p).relative_to(REPO_ROOT)))
            except ValueError:
                self._lut_var.set(p)

    #Changing dataset, especially checking for "Custom" selection to show/hide the custom dataset entry field
    def _on_dataset_change(self):
        if self._dataset_var.get() == "Custom…":
            self._custom_entry.pack(side="left", padx=(4, 0))
        else:
            self._custom_entry.pack_forget()
        self._refresh_code()

    #Add a new layer to the model
    def _add_row(self, initial_type=None):
        row = LayerRow(self._stack,
                       on_change=self._refresh_code,
                       on_remove=lambda r=None: self._remove_row(row),
                       initial_type=initial_type)
        row.pack(fill="x", pady=2, padx=2)
        self._rows.append(row)
        self._refresh_code()

    #Remove a layer from the model
    def _remove_row(self, row):
        if row in self._rows:
            self._rows.remove(row)
            row.destroy()
            self._refresh_code()

    # ── Code generation ───────────────────────────────────────────────────────

    #Generating the training script based on the user-selected parameters
    def _generate(self):
        cfgs     = [r.get_config() for r in self._rows]
        has_conv = any(c["type"] == "Conv2D" for c in cfgs)

        if has_conv:
            reshape = ("x_train = x_train.astype('float32') / 255.0\n"
                       "x_test  = x_test.astype('float32') / 255.0")
        else:
            reshape = ("x_train = x_train.reshape(x_train.shape[0], -1).astype('float32') / 255.0\n"
                       "x_test  = x_test.reshape(x_test.shape[0], -1).astype('float32') / 255.0")

        approx  = self._approx_var.get()
        lut     = self._lut_var.get().strip()

        layer_lines = "\n".join(
            "    " + self._layer_code(c, approx, lut) + "," for c in cfgs
        ) or "    # no layers defined"

        selected = self._dataset_var.get()
        dataset_id = self._custom_dataset_var.get().strip() if selected == "Custom…" else MAKER_DATASETS[selected]
        if not dataset_id:
            dataset_id = "mnist"

        try:
            lut_rel = str(Path(lut).relative_to(REPO_ROOT)) if lut else ""
        except ValueError:
            lut_rel = lut
        approx_imports = (
            "from python.keras.layers.amdenselayer import denseam\n"
            "from python.keras.layers.am_convolutional import AMConv2D\n"
            f"LUT = '{lut_rel}'\n"
        ) if approx else ""

        #Final generated script as a multiline string. Gets written to a temporary file and saved until next launch, in which case it will be deleted on next startup.
        return (
f"""import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
{approx_imports}

(x_train, y_train), (x_test, y_test) = tfds.as_numpy(tfds.load(
    '{dataset_id}', split=['train', 'test'], batch_size=-1, as_supervised=True))
{reshape}

model = tf.keras.Sequential([
{layer_lines}
])

model.compile(
    optimizer=tf.keras.optimizers.Adam({self._lr_var.get()}),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'],
)
model.fit(x_train, y_train, epochs={self._epochs_var.get()}, batch_size={self._batch_var.get()}, validation_split=0.1)
model.summary()
loss, acc = model.evaluate(x_test, y_test)
print(f"Test accuracy: {{acc:.4f}}")
"""
        )

    @staticmethod
    def _layer_code(cfg, approx=False, lut=""):
        t, p = cfg["type"], cfg["params"]
        if t == "Dense":
            if approx and lut:
                return f"denseam({p['units']}, activation='{p['activation']}', mant_mul_lut=LUT)"
            return f"tf.keras.layers.Dense({p['units']}, activation='{p['activation']}')"
        if t == "Conv2D":
            k = p["kernel_size"]
            if approx and lut:
                return (f"AMConv2D({p['filters']}, ({k}, {k}), "
                        f"activation='{p['activation']}', mant_mul_lut=LUT)")
            return (f"tf.keras.layers.Conv2D({p['filters']}, ({k}, {k}), "
                    f"activation='{p['activation']}')")
        if t == "Flatten":
            return "tf.keras.layers.Flatten()"
        if t == "Input (Flatten 28×28)":
            return "tf.keras.layers.Flatten(input_shape=x_train.shape[1:])"
        if t == "Dropout":
            return f"tf.keras.layers.Dropout({p['rate']})"
        if t == "MaxPooling2D":
            s = p["pool_size"]
            return f"tf.keras.layers.MaxPooling2D(({s}, {s}))"
        if t == "BatchNormalization":
            return "tf.keras.layers.BatchNormalization()"
        return f"# unknown layer: {t}"

    def _refresh_code(self):
        try:
            code = self._generate()
        except (ValueError, tk.TclError):
            return
        self._code_box.configure(state="normal")
        self._code_box.delete("1.0", tk.END)
        self._code_box.insert(tk.END, code)
        self._code_box.configure(state="disabled")

    # ── Run / save ────────────────────────────────────────────────────────────

    def _do_run(self):
        code = self._generate()
        tmp  = tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False,
            dir=REPO_ROOT, prefix="_approxtrain_maker_")
        tmp.write(code)
        tmp.close()
        self._run([sys.executable, "-u", tmp.name])

    def _save(self):
        path = filedialog.asksaveasfilename(
            title="Save generated script", defaultextension=".py",
            filetypes=[("Python files", "*.py")], initialdir=REPO_ROOT)
        if path:
            Path(path).write_text(self._generate())


# ─────────────────────────────────────────────────────────────────────────────
# Credits
# Pulls from credits.txt for easy editing and updating, especially for future contributions.
# It is encouraged for any future major contributors to add their name here.
# ─────────────────────────────────────────────────────────────────────────────

class CreditsFrame(tk.Frame):
    CREDITS_FILE = REPO_ROOT / "credits.txt"

    def __init__(self, parent, app):
        super().__init__(parent)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)
        self.rowconfigure(2, weight=3)
        ttk.Button(self, text="← Menu",
                   command=lambda: app.show_frame(MainMenuFrame)
                   ).grid(row=0, column=0, sticky="w", padx=12, pady=8)
        tk.Label(self, text="Credits",
                 font=app.fonts["page"]).grid(row=1, column=0, pady=(0, 8))

        text = self._load_credits()
        tk.Label(self, text=text, font=app.fonts["body"],
                 justify="center", wraplength=600).grid(row=2, column=0, padx=20)

    def _load_credits(self):
        try:
            with open(self.CREDITS_FILE, "r") as f:
                return f.read()
        except FileNotFoundError:
            return "(credits.txt not found)"

# ─────────────────────────────────────────────────────────────────────────────
# Options
# Used for changing font size for accessibility
# Also a placeholder for future options.
# ─────────────────────────────────────────────────────────────────────────────

class OptionsFrame(tk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent)
        self._app = app
        self.columnconfigure(0, weight=1)
        self.rowconfigure(2, weight=1)
        ttk.Button(self, text="← Menu",
                   command=lambda: app.show_frame(MainMenuFrame)
                   ).grid(row=0, column=0, sticky="w", padx=12, pady=8)
        tk.Label(self, text="Options",
                 font=app.fonts["page"]).grid(row=1, column=0, pady=(0, 20))

        settings = ttk.Frame(self)
        settings.grid(row=2, column=0, sticky="n")

        ttk.Label(settings, text="Font Size").grid(row=0, column=0, sticky="w", padx=12, pady=6)
        self._font_var = tk.IntVar(value=app.BASE_FONT_SIZE)
        scale = ttk.Scale(settings, from_=8, to=22, orient="horizontal",
                          variable=self._font_var, length=200,
                          command=lambda v: app.set_base_font_size(int(float(v))))
        scale.grid(row=0, column=1, padx=8)
        self._size_label = ttk.Label(settings, textvariable=self._font_var, width=3)
        self._size_label.grid(row=0, column=2, padx=(0, 12))

# ─────────────────────────────────────────────────────────────────────────────
# Main loop!
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    root = tk.Tk()
    app  = ApproxTrainGUI(root)
    root.mainloop()
