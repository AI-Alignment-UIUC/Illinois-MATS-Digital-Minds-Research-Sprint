"""SGLang server management + rollout client for the verbalization trainer.

The trainer owns an SGLang server process on the inference GPU, started with
LoRA support. After each optimizer step the freshly saved LoRA adapter is
synced by loading it under a versioned name (and unloading the previous
version); generation requests then reference the new name.
"""
import json
import subprocess
import sys
import time
from pathlib import Path

import requests


class SGLangServer:
    def __init__(self, python: str, model_path: str, gpu_id: int, port: int, *,
                 mem_fraction: float = 0.85, max_lora_rank: int = 8,
                 lora_target_modules: tuple[str, ...] = ("gate_proj", "up_proj", "down_proj"),
                 log_path: str | None = None):
        self.base = f"http://127.0.0.1:{port}"
        self.port = port
        # fail fast on a squatted port: a half-dead server there would make
        # the launch fail at bind and the health wait time out confusingly
        import socket
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", port)) == 0:
                raise RuntimeError(
                    f"port {port} is already in use (stale SGLang server?). "
                    f"Kill it (pgrep -af sglang) or change run.sglang_port.")
        cmd = [
            python, "-m", "sglang.launch_server",
            "--model-path", model_path,
            "--host", "127.0.0.1", "--port", str(port),
            "--mem-fraction-static", str(mem_fraction),
            "--enable-lora",
            "--max-lora-rank", str(max_lora_rank),
            "--lora-target-modules", *lora_target_modules,
        ]
        env = {"CUDA_VISIBLE_DEVICES": str(gpu_id),
               # flashinfer-cubin has no matching release for flashinfer
               # 0.6.15; the mismatch is benign (cubins load lazily)
               "FLASHINFER_DISABLE_VERSION_CHECK": "1"}
        import os
        env = {**os.environ, **env}
        self.log_fh = open(log_path, "w") if log_path else subprocess.DEVNULL
        self.proc = subprocess.Popen(cmd, stdout=self.log_fh, stderr=subprocess.STDOUT, env=env)
        self.lora_version = 0
        self.lora_name: str | None = None

    def wait_ready(self, timeout: float = 1200.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(
                    f"sglang server exited with code {self.proc.returncode}; see its log")
            try:
                r = requests.get(f"{self.base}/health", timeout=5)
                if r.status_code == 200:
                    return
            except requests.RequestException:
                pass
            time.sleep(2)
        raise TimeoutError("sglang server did not become healthy in time")

    def sync_lora(self, adapter_dir: str | Path):
        """Load the adapter under a fresh versioned name; unload the previous."""
        if not hasattr(self, "lora_prefix"):
            import os
            self.lora_prefix = f"verbalizer_{os.getpid()}"
        new_name = f"{self.lora_prefix}_v{self.lora_version + 1}"
        r = requests.post(f"{self.base}/load_lora_adapter", json={
            "lora_name": new_name,
            "lora_path": str(adapter_dir),
        }, timeout=120)
        if r.status_code != 200:
            raise RuntimeError(f"load_lora_adapter failed: {r.status_code} {r.text[:500]}")
        old_name = self.lora_name
        self.lora_name = new_name
        self.lora_version += 1
        if old_name is not None:
            requests.post(f"{self.base}/unload_lora_adapter",
                          json={"lora_name": old_name}, timeout=120)

    def generate(self, prompts: list[str], *, group_size: int, max_new_tokens: int,
                 temperature: float = 1.0, stop: list[str] | None = None
                 ) -> tuple[list[str], list[list[int]], list[bool]]:
        """Sample group_size completions per prompt. Returns flat prompt-major
        lists of completion strings, their exact sampled token ids (including
        the stop token, so the policy learns to stop), and a truncated flag
        per row (hit max_new_tokens without finishing)."""
        expanded = [p for p in prompts for _ in range(group_size)]
        payload = {
            "text": expanded,
            "sampling_params": {
                "temperature": temperature,
                "max_new_tokens": max_new_tokens,
                "top_p": 1.0,
                **({"stop": stop} if stop else {}),
            },
            "return_logprob": True,
        }
        if self.lora_name is not None:
            payload["lora_path"] = [self.lora_name] * len(expanded)
        r = requests.post(f"{self.base}/generate", json=payload, timeout=1800)
        if r.status_code != 200:
            raise RuntimeError(f"generate failed: {r.status_code} {r.text[:500]}")
        out = r.json()
        texts = [row["text"] for row in out]
        ids = [[t[1] for t in row["meta_info"]["output_token_logprobs"]]
               for row in out]
        truncated = [row["meta_info"].get("finish_reason", {}).get("type") == "length"
                     for row in out]
        return texts, ids, truncated

    def flush_cache(self):
        try:
            requests.post(f"{self.base}/flush_cache", timeout=60)
        except requests.RequestException:
            pass

    def shutdown(self):
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        if self.log_fh is not subprocess.DEVNULL:
            self.log_fh.close()
