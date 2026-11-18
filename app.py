"""LLVC Voice Changer - Windows GUI. Run: python app.py"""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import json, os
import sounddevice as sd
from engine import PassthroughEngine, LLVCEngine
from pipeline import Pipeline

CFG = os.path.join(os.path.expanduser("~"), ".llvc_app.json")


def devices(kind):
    out = []
    for i, d in enumerate(sd.query_devices()):
        if d[f"max_{kind}_channels"] > 0:
            api = sd.query_hostapis(d["hostapi"])["name"].replace("Windows ", "")
            out.append((f"{d['name']}  [{api}]", i))
    out.sort(key=lambda t: (0 if "WASAPI" in t[0] else 1, t[0]))
    return out


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("LLVC Voice Changer")
        self.geometry("560x520")
        self.pipe = None
        self.cfg = json.load(open(CFG)) if os.path.exists(CFG) else {}
        ins, outs = devices("input"), devices("output")
        self.in_map, self.out_map = dict(ins), dict(outs)
        pad = dict(padx=10, pady=4, sticky="we")
        f = ttk.Frame(self); f.pack(fill="both", expand=True, padx=8, pady=8); f.columnconfigure(1, weight=1)

        self.v = {k: tk.StringVar(value=self.cfg.get(k, "")) for k in ("in", "out", "mon", "repo", "ckpt", "conf")}
        self.mode = tk.StringVar(value=self.cfg.get("mode", "test"))
        self.chunk = tk.IntVar(value=self.cfg.get("chunk", 1))
        self.gain = tk.DoubleVar(value=1.0)

        def combo(row, label, key, values, optional=False):
            ttk.Label(f, text=label).grid(row=row, column=0, sticky="w", padx=10)
            vals = ([""] if optional else []) + [n for n, _ in values]
            c = ttk.Combobox(f, textvariable=self.v[key], values=vals, state="readonly"); c.grid(row=row, column=1, columnspan=2, **pad)
            if not self.v[key].get() and vals: c.current(0)
        combo(0, "Microphone (input)", "in", ins)
        combo(1, "Output -> 'CABLE Input'", "out", outs)
        combo(2, "Monitor (your headphones)", "mon", outs, optional=True)

        ttk.Label(f, text="Mode").grid(row=3, column=0, sticky="w", padx=10)
        m = ttk.Frame(f); m.grid(row=3, column=1, columnspan=2, sticky="w", padx=10)
        ttk.Radiobutton(m, text="Test (no conversion)", variable=self.mode, value="test").pack(side="left")
        ttk.Radiobutton(m, text="LLVC", variable=self.mode, value="llvc").pack(side="left", padx=12)

        def path(row, label, key, folder=False):
            ttk.Label(f, text=label).grid(row=row, column=0, sticky="w", padx=10)
            ttk.Entry(f, textvariable=self.v[key]).grid(row=row, column=1, **pad)
            ttk.Button(f, text="...", width=3, command=lambda: self.v[key].set(
                (filedialog.askdirectory() if folder else filedialog.askopenfilename()) or self.v[key].get())).grid(row=row, column=2, padx=6)
        path(4, "LLVC repo folder", "repo", True)
        path(5, "Checkpoint (.pth)", "ckpt")
        path(6, "Config (.json)", "conf")

        ttk.Label(f, text="Chunk factor (higher = safer, more delay)").grid(row=7, column=0, sticky="w", padx=10)
        ttk.Spinbox(f, from_=1, to=16, textvariable=self.chunk, width=5).grid(row=7, column=1, sticky="w", padx=10)
        ttk.Label(f, text="Output gain").grid(row=8, column=0, sticky="w", padx=10)
        ttk.Scale(f, from_=0.2, to=3.0, variable=self.gain, command=self._gain).grid(row=8, column=1, columnspan=2, **pad)

        self.bi, self.bo = ttk.Progressbar(f, maximum=1.0), ttk.Progressbar(f, maximum=1.0)
        ttk.Label(f, text="Input level").grid(row=9, column=0, sticky="w", padx=10); self.bi.grid(row=9, column=1, columnspan=2, **pad)
        ttk.Label(f, text="Output level").grid(row=10, column=0, sticky="w", padx=10); self.bo.grid(row=10, column=1, columnspan=2, **pad)

        self.btn = ttk.Button(f, text="Start", command=self.toggle); self.btn.grid(row=11, column=0, columnspan=3, pady=10)
        self.status = ttk.Label(f, text="Stopped.", wraplength=520); self.status.grid(row=12, column=0, columnspan=3, padx=10, sticky="w")
        ttk.Label(f, foreground="#555", wraplength=520, text="In WhatsApp Desktop: Settings > Voice and video > Microphone = 'CABLE Output'. "
                  "Only convert voices you have permission to use.").grid(row=13, column=0, columnspan=3, padx=10, pady=8, sticky="w")
        self.protocol("WM_DELETE_WINDOW", self.close)
        self.after(100, self.tick)

    def _gain(self, _=None):
        if self.pipe: self.pipe.gain = self.gain.get()

    def toggle(self):
        if self.pipe: return self.stop()
        try:
            if self.mode.get() == "llvc":
                r, c, j = self.v["repo"].get(), self.v["ckpt"].get(), self.v["conf"].get()
                if not (r and c and j): raise ValueError("Pick the LLVC repo folder, checkpoint and config.")
                eng = LLVCEngine(r, c, j, self.chunk.get())
            else:
                eng = PassthroughEngine(640 * self.chunk.get())
            mon = self.out_map.get(self.v["mon"].get())
            self.pipe = Pipeline(eng, self.in_map[self.v["in"].get()], self.out_map[self.v["out"].get()], mon, self.gain.get())
            self.pipe.start()
        except Exception as e:
            self.pipe = None
            messagebox.showerror("Could not start", str(e)); return
        self.btn.config(text="Stop")
        self.lat = 2 * eng.chunk / eng.sr * 1000
        json.dump({**{k: x.get() for k, x in self.v.items()}, "mode": self.mode.get(), "chunk": self.chunk.get()}, open(CFG, "w"))

    def stop(self):
        if self.pipe: self.pipe.stop(); self.pipe = None
        self.btn.config(text="Start"); self.status.config(text="Stopped."); self.bi["value"] = self.bo["value"] = 0

    def tick(self):
        p = self.pipe
        if p:
            self.bi["value"], self.bo["value"] = min(p.level_in, 1), min(p.level_out, 1)
            warn = "  SLOWER THAN REAL TIME - raise chunk factor" if p.rtf > 0.9 else ""
            self.status.config(text=f"Running. ~{self.lat:.0f} ms buffering + compute | load {p.rtf*100:.0f}% | dropped {p.drops} | underruns {p.underruns}{warn}")
        self.after(100, self.tick)

    def close(self):
        self.stop(); self.destroy()


if __name__ == "__main__":
    App().mainloop()
