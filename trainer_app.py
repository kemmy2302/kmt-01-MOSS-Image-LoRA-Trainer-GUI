from __future__ import annotations

import json
import mimetypes
import os
import re
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, urlparse


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = Path(os.environ.get("MOSS_TRAINER_CONFIG", ROOT / "config.json"))
if not CONFIG_PATH.is_file():
    CONFIG_PATH = ROOT / "config.example.json"
APP_CONFIG = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def _configured_path(key: str, default: str) -> Path:
    value = os.path.expandvars(str(APP_CONFIG.get(key, default)))
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


SD_SCRIPTS = _configured_path("sd_scripts", "sd-scripts")
PYTHON = _configured_path("python", ".venv/Scripts/python.exe")
ACCELERATE = PYTHON.parent / "accelerate.exe"
RUNTIME = ROOT / "runtime"
CONFIGS = ROOT / "configs"
WEB_DIR = ROOT / "web"
OUTPUT_DIR = _configured_path("output_dir", "output")
DIT = _configured_path("base_model", "models/moss_image_base_int8_convrot.safetensors")
TEXT_ENCODER = _configured_path("text_encoder", "models/qwen3.5_9b_int8_convrot.safetensors")
VAE = _configured_path("vae", "models/flux2-vae_small.safetensors")
STATE_PATH = RUNTIME / "training-state.json"
LOG_PATH = RUNTIME / "training.log"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
MODEL_MIN_BYTES = {DIT: 5_400_000_000, TEXT_ENCODER: 9_700_000_000, VAE: 240_000_000}

RUNTIME.mkdir(parents=True, exist_ok=True)
CONFIGS.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

_lock = threading.RLock()
_process: subprocess.Popen | None = None
_tagger_process: subprocess.Popen | None = None
TAGGER_LOG_PATH = RUNTIME / "tagger.log"


def _safe_name(value: str) -> str:
    value = re.sub(r"[^0-9A-Za-z._-]+", "_", value.strip()).strip("._")
    if not value:
        raise ValueError("出力名を入力してください（英数字・_・-を推奨）。")
    return value[:120]


def _toml_string(path: Path) -> str:
    return json.dumps(path.resolve().as_posix(), ensure_ascii=False)


def _dataset_pairs(directory: Path) -> tuple[int, list[str]]:
    images = [p for p in directory.iterdir() if p.is_file() and p.suffix.casefold() in IMAGE_SUFFIXES]
    missing = [p.name for p in images if not p.with_suffix(".txt").is_file()
               or not p.with_suffix(".txt").read_text(encoding="utf-8", errors="replace").strip()]
    return len(images) - len(missing), missing


def _read_state() -> dict:
    try:
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"status": "idle", "message": "学習は開始されていません。"}


def _write_state(**updates) -> dict:
    state = _read_state()
    state.update(updates, updated_at=time.strftime("%Y-%m-%d %H:%M:%S"))
    STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    return state


def _latest_progress() -> str:
    try:
        with LOG_PATH.open("rb") as stream:
            stream.seek(0, os.SEEK_END)
            stream.seek(max(0, stream.tell() - 160_000))
            text = stream.read().decode("utf-8", errors="replace")
    except OSError:
        return ""
    text = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", text)
    lines = [line.strip() for line in text.replace("\r", "\n").splitlines() if line.strip()]
    for line in reversed(lines):
        if "steps:" in line or re.search(r"\b\d{1,3}%\|.*\|\s*\d+/\d+", line):
            return line[-1000:]
    return lines[-1][-1000:] if lines else ""


def _log_tail(limit: int = 30000) -> str:
    try:
        with LOG_PATH.open("rb") as stream:
            stream.seek(0, os.SEEK_END)
            stream.seek(max(0, stream.tell() - limit))
            text = stream.read().decode("utf-8", errors="replace")
    except OSError:
        return ""
    return re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", text).replace("\r", "\n")[-limit:]


def status() -> dict:
    with _lock:
        state = _read_state()
        running = _process is not None and _process.poll() is None
        if state.get("status") == "running" and not running:
            pid = state.get("pid")
            if isinstance(pid, int):
                try:
                    import psutil
                    process = psutil.Process(pid)
                    command = " ".join(process.cmdline()).casefold()
                    running = "moss_image_train_network.py" in command
                except (OSError, psutil.Error):
                    pass
        state["running"] = running
        state["progress"] = _latest_progress()
        state["log_tail"] = _log_tail()
        state["models"] = {
            "base": DIT.is_file() and DIT.stat().st_size >= MODEL_MIN_BYTES[DIT],
            "text_encoder": TEXT_ENCODER.is_file() and TEXT_ENCODER.stat().st_size >= MODEL_MIN_BYTES[TEXT_ENCODER],
            "vae": VAE.is_file() and VAE.stat().st_size >= MODEL_MIN_BYTES[VAE],
        }
        return state


def _watch_process(proc: subprocess.Popen, output_name: str) -> None:
    global _process
    code = proc.wait()
    output = OUTPUT_DIR / f"{output_name}.safetensors"
    with _lock:
        current = _read_state()
        if current.get("status") == "stopping":
            final = "stopped"
        else:
            final = "completed" if code == 0 else "failed"
        messages = {
            "completed": "学習が正常に完了しました。",
            "failed": "学習がエラー終了しました。ログを確認してください。",
            "stopped": "学習を停止しました。",
        }
        _write_state(status=final, message=messages[final], exit_code=code,
                     output_file=str(output) if output.is_file() else None)
        _process = None


def start_training(payload: dict) -> dict:
    global _process
    with _lock:
        if _process is not None and _process.poll() is None:
            raise RuntimeError("すでに学習中です。")
        missing_models = [str(p) for p in (DIT, TEXT_ENCODER, VAE) if not p.is_file() or p.stat().st_size < MODEL_MIN_BYTES[p]]
        if missing_models:
            raise RuntimeError("必要モデルが不足しています: " + ", ".join(missing_models))
        dataset_dir = Path(str(payload.get("dataset_dir", "")).strip()).expanduser()
        if not dataset_dir.is_absolute() or not dataset_dir.is_dir():
            raise ValueError("存在する学習画像フォルダの絶対パスを指定してください。")
        pair_count, missing_captions = _dataset_pairs(dataset_dir)
        if missing_captions:
            preview = ", ".join(missing_captions[:8])
            raise ValueError(f"同名.txtがない画像があります: {preview}")
        if pair_count == 0:
            raise ValueError("画像と同名.txtの組を1件以上用意してください。")

        output_name = _safe_name(str(payload.get("output_name", "")))
        resolution = int(payload.get("resolution", 1024))
        training_epochs = int(payload.get("training_epochs", 10))
        if training_epochs not in {5, 10, 20}:
            raise ValueError("学習強度は5、10、20から選択してください。")
        # 操作を単純化するため、固定10 repeats相当 × 選択epoch数で計算する。
        max_steps = pair_count * 10 * training_epochs
        save_every = min(int(payload.get("save_every", 250)), max_steps)
        rank = int(payload.get("rank", 16))
        alpha = int(payload.get("alpha", 16))
        blocks = int(payload.get("blocks_to_swap", 24))
        learning_rate = float(payload.get("learning_rate", 0.0001))
        if resolution not in {512, 768, 1024}:
            raise ValueError("解像度は512、768、1024から選択してください。")
        if not 1 <= max_steps <= 100000 or not 1 <= save_every <= max_steps:
            raise ValueError("ステップ数または途中保存間隔が不正です。")
        if rank not in {4, 8, 16, 32, 64} or alpha < 1 or alpha > rank:
            raise ValueError("rank/alpha設定が不正です。")
        if not 20 <= blocks <= 36:
            raise ValueError("RTX 4070向けblocks_to_swapは20～36で指定してください。")

        config_path = CONFIGS / f"{output_name}.dataset.toml"
        config_path.write_text(
            "[general]\n"
            "caption_extension = \".txt\"\n"
            "enable_multiline_caption = true\n"
            "shuffle_caption = false\n\n"
            "[[datasets]]\n"
            f"resolution = {resolution}\n"
            "batch_size = 1\n"
            "enable_bucket = true\n"
            "bucket_no_upscale = true\n"
            "bucket_reso_steps = 64\n\n"
            "  [[datasets.subsets]]\n"
            f"  image_dir = {_toml_string(dataset_dir)}\n"
            "  num_repeats = 1\n",
            encoding="utf-8",
        )
        argv = [
            str(ACCELERATE), "launch", "--num_cpu_threads_per_process", "1",
            "--mixed_precision", "bf16", "--num_processes", "1", "--num_machines", "1",
            "--dynamo_backend", "no",
            str(SD_SCRIPTS / "moss_image_train_network.py"),
            f"--pretrained_model_name_or_path={DIT}", f"--text_encoder={TEXT_ENCODER}", f"--vae={VAE}",
            f"--dataset_config={config_path}", f"--output_dir={OUTPUT_DIR}", f"--output_name={output_name}",
            "--save_model_as=safetensors", "--network_module=networks.lora_generic",
            f"--network_dim={rank}", f"--network_alpha={alpha}", "--network_train_unet_only",
            f"--learning_rate={learning_rate}", "--optimizer_type=AdamW8bit", "--lr_scheduler=constant",
            "--timestep_sampling=sigma", "--weighting_scheme=logit_normal", "--logit_mean=-1.09",
            "--logit_std=0.95", "--discrete_flow_shift=1.0", f"--max_train_steps={max_steps}",
            f"--save_every_n_steps={save_every}", "--mixed_precision=bf16", "--gradient_checkpointing",
            "--cache_latents", "--cache_latents_to_disk", "--cache_text_encoder_outputs",
            "--cache_text_encoder_outputs_to_disk", f"--blocks_to_swap={blocks}",
            "--max_data_loader_n_workers=2", "--persistent_data_loader_workers", "--seed=42",
        ]
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        log = LOG_PATH.open("ab", buffering=0)
        log.write(f"\n--- start {time.strftime('%Y-%m-%d %H:%M:%S')} {output_name} ---\n".encode())
        flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        _process = subprocess.Popen(argv, cwd=SD_SCRIPTS, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, creationflags=flags)
        _write_state(status="running", message="学習を実行中です。", pid=_process.pid,
                     exit_code=None, output_file=None,
                     output_name=output_name, dataset_dir=str(dataset_dir),
                     pair_count=pair_count, training_epochs=training_epochs, max_steps=max_steps,
                     log=str(LOG_PATH), config=str(config_path))
        threading.Thread(target=_watch_process, args=(_process, output_name), daemon=True).start()
        return status()


def stop_training() -> dict:
    global _process
    with _lock:
        if _process is None or _process.poll() is not None:
            return status()
        _write_state(status="stopping")
        try:
            import psutil
            root = psutil.Process(_process.pid)
            targets = root.children(recursive=True) + [root]
            for proc in reversed(targets):
                try:
                    proc.terminate()
                except psutil.Error:
                    pass
        except Exception:
            _process.terminate()
    return status()


def tagger_status() -> dict:
    running = _tagger_process is not None and _tagger_process.poll() is None
    try:
        lines = TAGGER_LOG_PATH.read_text(encoding="utf-8", errors="replace").splitlines()
        detail = lines[-1] if lines else ""
    except OSError:
        detail = ""
    return {"running": running, "exit_code": None if running or _tagger_process is None else _tagger_process.returncode,
            "progress": detail, "log": str(TAGGER_LOG_PATH)}


def start_tagger(payload: dict) -> dict:
    global _tagger_process
    if _tagger_process is not None and _tagger_process.poll() is None:
        raise RuntimeError("タグ抽出はすでに実行中です。")
    if _process is not None and _process.poll() is None:
        raise RuntimeError("学習中はタグ抽出を開始できません。")
    directory = Path(str(payload.get("dataset_dir", "")).strip())
    if not directory.is_absolute() or not directory.is_dir():
        raise ValueError("存在する画像フォルダの絶対パスを指定してください。")
    threshold = float(payload.get("tag_threshold", 0.35))
    if not 0.1 <= threshold <= 0.95:
        raise ValueError("タグ閾値は0.1～0.95で指定してください。")
    argv = [str(PYTHON), str(ROOT / "moss_tagger.py"), "--dataset-dir", str(directory),
            "--threshold", str(threshold), "--trigger", str(payload.get("trigger_word", "")).strip()]
    if bool(payload.get("overwrite_captions", False)):
        argv.append("--overwrite")
    log = TAGGER_LOG_PATH.open("wb", buffering=0)
    _tagger_process = subprocess.Popen(argv, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log,
                                       stderr=subprocess.STDOUT,
                                       creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0))
    return tagger_status()


def _safe_dataset_file(directory_value: str, name: str) -> tuple[Path, Path]:
    directory = Path(directory_value).resolve()
    if not directory.is_dir():
        raise ValueError("画像フォルダが見つかりません。")
    if Path(name).name != name:
        raise ValueError("不正なファイル名です。")
    image = (directory / name).resolve()
    if image.parent != directory or not image.is_file() or image.suffix.casefold() not in IMAGE_SUFFIXES:
        raise ValueError("対象画像が見つかりません。")
    return directory, image


def dataset_items(directory_value: str) -> list[dict]:
    directory = Path(directory_value).resolve()
    if not directory.is_dir():
        raise ValueError("画像フォルダが見つかりません。")
    result = []
    for image in sorted(directory.iterdir(), key=lambda item: item.name.casefold()):
        if not image.is_file() or image.suffix.casefold() not in IMAGE_SUFFIXES:
            continue
        caption = image.with_suffix(".txt")
        result.append({"name": image.name, "caption": caption.read_text(encoding="utf-8", errors="replace") if caption.is_file() else ""})
    return result


def save_caption(payload: dict) -> dict:
    _, image = _safe_dataset_file(str(payload.get("dataset_dir", "")), str(payload.get("name", "")))
    text = str(payload.get("caption", "")).replace("\r\n", "\n").replace("\r", "\n").strip()
    image.with_suffix(".txt").write_text(text, encoding="utf-8")
    return {"saved": True, "name": image.name}


def bulk_tags(payload: dict) -> dict:
    directory = Path(str(payload.get("dataset_dir", ""))).resolve()
    if not directory.is_dir():
        raise ValueError("画像フォルダが見つかりません。")
    action = str(payload.get("action", ""))
    requested = [tag.strip() for tag in str(payload.get("tags", "")).split(",") if tag.strip()]
    if action not in {"prepend", "remove"} or not requested:
        raise ValueError("追加または削除するタグを入力してください。")
    changed = 0
    for image in directory.iterdir():
        if not image.is_file() or image.suffix.casefold() not in IMAGE_SUFFIXES:
            continue
        caption_path = image.with_suffix(".txt")
        existing = caption_path.read_text(encoding="utf-8", errors="replace") if caption_path.is_file() else ""
        tags = [tag.strip() for tag in existing.split(",") if tag.strip()]
        before = list(tags)
        if action == "prepend":
            present = {tag.casefold() for tag in tags}
            additions = [tag for tag in requested if tag.casefold() not in present]
            tags = additions + tags
        else:
            removal = {tag.casefold() for tag in requested}
            tags = [tag for tag in tags if tag.casefold() not in removal]
        if tags != before:
            caption_path.write_text(", ".join(tags), encoding="utf-8")
            changed += 1
    return {"changed_files": changed, "action": action, "tags": requested}




class Handler(BaseHTTPRequestHandler):
    def _json(self, data: dict, code: int = 200) -> None:
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(code); self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        if path in {"/", "/index.html"}:
            page = WEB_DIR / "index.html"
            body = page.read_bytes() if page.is_file() else b"GUI file not found: web/index.html"
            self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
        elif path in {"/api/status", "/health"}:
            self._json(status())
        elif path == "/api/tagger-status":
            self._json(tagger_status())
        elif path == "/api/dataset":
            query = parse_qs(parsed.query)
            items = dataset_items(query.get("directory", [""])[0])
            self._json({"items": items, "count": len(items), "missing_captions": sum(not item["caption"].strip() for item in items)})
        elif path == "/api/dataset-image":
            query = parse_qs(parsed.query)
            _, image = _safe_dataset_file(query.get("directory", [""])[0], query.get("name", [""])[0])
            body = image.read_bytes()
            self.send_response(200); self.send_header("Content-Type", mimetypes.guess_type(image.name)[0] or "application/octet-stream")
            self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self) -> None:
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length) or b"{}")
            path = urlparse(self.path).path
            if path == "/api/start": self._json(start_training(payload))
            elif path == "/api/stop": self._json(stop_training())
            elif path == "/api/run-tagger": self._json(start_tagger(payload))
            elif path == "/api/save-caption": self._json(save_caption(payload))
            elif path == "/api/bulk-tags": self._json(bulk_tags(payload))
            else: self._json({"error": "not found"}, 404)
        except Exception as exc:
            self._json({"error": str(exc)}, 400)

    def log_message(self, fmt: str, *args) -> None:
        print(f"[{self.log_date_time_string()}] {fmt % args}", flush=True)


if __name__ == "__main__":
    listen_host = str(APP_CONFIG.get("listen_host", "127.0.0.1"))
    port = int(APP_CONFIG.get("port", 8004))
    server = ThreadingHTTPServer((listen_host, port), Handler)
    print(f"MOSS-Image LoRA Trainer: http://{listen_host}:{port}", flush=True)
    server.serve_forever()
