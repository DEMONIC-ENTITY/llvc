"""Mic -> engine -> VB-Cable (+ optional monitor). Three PortAudio streams + one worker thread."""
import queue, threading, time
import numpy as np
import sounddevice as sd


class Pipeline:
    def __init__(self, engine, in_dev, out_dev, mon_dev=None, gain=1.0):
        self.e, self.gain = engine, gain
        self.in_dev, self.out_dev, self.mon_dev = in_dev, out_dev, mon_dev
        self.inq, self.outq, self.monq = queue.Queue(8), queue.Queue(8), queue.Queue(8)
        self.level_in = self.level_out = 0.0
        self.rtf = 0.0
        self.drops = self.underruns = 0
        self._run = False
        self._streams = []
        self._buf = np.zeros(0, np.float32)

    def _extra(self, dev):
        info = sd.query_devices(dev)
        api = sd.query_hostapis(info["hostapi"])["name"]
        return sd.WasapiSettings(auto_convert=True) if "WASAPI" in api else None

    def _in_cb(self, data, frames, t, status):
        try:
            self.inq.put_nowait(data[:, 0].copy())
        except queue.Full:
            self.drops += 1

    def _make_out_cb(self, q):
        def cb(outdata, frames, t, status):
            try:
                outdata[:, 0] = q.get_nowait()[:frames]
            except queue.Empty:
                outdata.fill(0)
                self.underruns += 1
        return cb

    def _worker(self):
        n = self.e.chunk
        while self._run:
            try:
                x = self.inq.get(timeout=0.2)
            except queue.Empty:
                continue
            self._buf = np.concatenate([self._buf, x])
            while len(self._buf) >= n:
                blk, self._buf = self._buf[:n], self._buf[n:]
                t0 = time.perf_counter()
                y = self.e.process(blk) * self.gain
                y = np.clip(y, -1.0, 1.0).astype(np.float32)
                self.rtf = (time.perf_counter() - t0) / (n / self.e.sr)
                self.level_in = float(np.abs(blk).max())
                self.level_out = float(np.abs(y).max())
                for q in ([self.outq] + ([self.monq] if self.mon_dev is not None else [])):
                    try:
                        q.put_nowait(y)
                    except queue.Full:
                        pass

    def start(self):
        sr, n = self.e.sr, self.e.chunk
        self._run = True
        self._streams = [
            sd.InputStream(device=self.in_dev, samplerate=sr, channels=1, blocksize=n,
                           dtype="float32", callback=self._in_cb, extra_settings=self._extra(self.in_dev)),
            sd.OutputStream(device=self.out_dev, samplerate=sr, channels=1, blocksize=n,
                            dtype="float32", callback=self._make_out_cb(self.outq),
                            extra_settings=self._extra(self.out_dev)),
        ]
        if self.mon_dev is not None:
            self._streams.append(sd.OutputStream(device=self.mon_dev, samplerate=sr, channels=1, blocksize=n,
                                 dtype="float32", callback=self._make_out_cb(self.monq),
                                 extra_settings=self._extra(self.mon_dev)))
        threading.Thread(target=self._worker, daemon=True).start()
        for s in self._streams:
            s.start()

    def stop(self):
        self._run = False
        for s in self._streams:
            try:
                s.stop(); s.close()
            except Exception:
                pass
        self._streams = []
