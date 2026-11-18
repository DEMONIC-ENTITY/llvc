"""Voice engines. Each engine exposes: sr (int), chunk (int samples), process(np.float32[chunk]) -> np.float32[chunk]."""
import os, sys, json
import numpy as np


class PassthroughEngine:
    """Test engine: no conversion. Use it to verify mic -> VB-Cable -> WhatsApp routing first."""
    sr = 16000

    def __init__(self, chunk=640):
        self.chunk = chunk

    def process(self, x):
        return x


class LLVCEngine:
    """Adapter around the official LLVC repo (github.com/KoeAI/LLVC).

    UNVERIFIED against the repo source: the streaming calls below follow infer.py's
    load_model() / do_infer() loop as I recall it. If you get an AttributeError or a
    shape error, open infer.py in the LLVC folder and match the calls in _load()
    and process() to what do_infer() does. Nothing else in the app needs to change.
    """

    def __init__(self, repo_dir, checkpoint, config, chunk_factor=1, threads=0):
        import torch
        self.torch = torch
        if threads:
            torch.set_num_threads(threads)
        if repo_dir not in sys.path:
            sys.path.insert(0, repo_dir)
        os.chdir(repo_dir)  # repo uses relative imports/paths
        self._load(checkpoint, config)
        m = self.model
        for attr in ("dec_chunk_size", "L", "init_buffers"):
            if not hasattr(m, attr):
                raise RuntimeError(f"Model has no '{attr}'. Compare with do_infer() in LLVC/infer.py.")
        self.chunk = int(m.dec_chunk_size * m.L * chunk_factor)
        self._reset()

    def _load(self, checkpoint, config):
        from infer import load_model  # from the LLVC repo
        self.model, self.sr = load_model(config, checkpoint) if self._cfg_first() else load_model(checkpoint, config)
        self.model.eval()

    @staticmethod
    def _cfg_first():
        # infer.py's argument order may differ between forks; flip to True if load_model(config, checkpoint).
        return False

    def _reset(self):
        t, m = self.torch, self.model
        dev = "cpu"
        self.enc_buf, self.dec_buf, self.out_buf = m.init_buffers(1, dev)
        self.ctx = m.convnet_pre.init_ctx_buf(1, dev) if hasattr(m, "convnet_pre") else None

    def process(self, x):
        t, m = self.torch, self.model
        with t.no_grad():
            inp = t.from_numpy(x).float().reshape(1, 1, -1)
            out, self.enc_buf, self.dec_buf, self.out_buf, self.ctx = m(
                inp, self.enc_buf, self.dec_buf, self.out_buf, self.ctx,
                pad=(not getattr(m, "lookahead", False)))
        return out.reshape(-1).cpu().numpy().astype(np.float32)
