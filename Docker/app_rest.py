import os, io, time, threading, logging, pickle, random
from typing import Any, Optional
from collections import deque

import numpy as np
from flask import Flask, render_template, jsonify, request, Response, send_from_directory

import gymnasium as gym
from PIL import Image

# ----------------- Config -----------------
ENV_ID = os.environ.get("ENV_ID", "LunarLander-v2")     # Gymnasium 0.29 => v2
FPS     = int(os.environ.get("FPS", "20"))              # tope de frames/seg
STEP_DELAY_MS = int(os.environ.get("STEP_DELAY_MS", "50"))  # pausa extra por paso (ms)

HOST    = os.environ.get("HOST", "0.0.0.0")
PORT    = int(os.environ.get("PORT", "8000"))

MODELS_DIR = os.environ.get("MODELS_DIR", "/app/models")
S3_BUCKET  = os.environ.get("S3_BUCKET", "").strip()
S3_PREFIX  = os.environ.get("S3_PREFIX", "").strip()
AUTO_PULL_S3 = bool(int(os.environ.get("AUTO_PULL_S3", "0")))

# grabación opcional de episodios de modelos
RECORD_VIDEOS = bool(int(os.environ.get("RECORD_VIDEOS", "0")))
VIDEO_DIR = os.environ.get("VIDEO_DIR", "/app/videos")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ----------------- App -----------------
app = Flask(__name__, template_folder="templates")

estado = {
    "en_ejecucion": False,
    "modo": "idle",                # idle|humano|demo|modelo
    "modelo": None,
    "recompensa_total": 0.0,
    "episodio": 0,
}

stop_flag   = threading.Event()
frame_lock  = threading.Lock()
last_jpeg   = None
keys_state  = {"left": False, "main": False, "right": False}

model_cache = {}

# ----------------- Util -----------------
def to_jpeg(rgb_array: np.ndarray) -> bytes:
    im = Image.fromarray(rgb_array)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=85)
    return buf.getvalue()

def make_env():
    env = gym.make(ENV_ID, render_mode="rgb_array")
    return env

def maybe_wrap_record(env, prefix: str):
    if not RECORD_VIDEOS:
        return env
    try:
        from gymnasium.wrappers import RecordVideo
        os.makedirs(VIDEO_DIR, exist_ok=True)
        logging.info(f"[video] grabando episodios en {VIDEO_DIR} (prefijo={prefix})")
        return RecordVideo(env, video_folder=VIDEO_DIR,
                           episode_trigger=lambda ep: True,
                           name_prefix=prefix)
    except Exception as e:
        logging.warning(f"No pude activar RecordVideo: {e}")
        return env

def decide_action_from_keys(keys: dict) -> int:
    # 0: noop, 1: left, 2: main, 3: right
    if keys.get("main"):  return 2
    if keys.get("left"):  return 1
    if keys.get("right"): return 3
    return 0

def _unwrap_possible_container(modelo):
    # Si viene como dict {'agent':obj} o {'policy':obj}…
    if isinstance(modelo, dict):
        for k in ("agent", "model", "policy"):
            if k in modelo:
                logging.info(f"Desempaquetado modelo a través de dict['{k}']")
                return modelo[k]
    return modelo

def decide_action_from_model(modelo: Any, obs: np.ndarray) -> int:
    # Greedy si el agente trae epsilon
    try:
        if hasattr(modelo, "epsilon"):
            setattr(modelo, "epsilon", 0.0)
    except Exception:
        pass

    # SB3
    if hasattr(modelo, "predict"):
        act, _ = modelo.predict(obs, deterministic=True)
        if isinstance(act, (list, np.ndarray)):
            act = int(act[0])
        return int(act)

    # Otros
    for attr in ("act", "choose_action"):
        if hasattr(modelo, attr):
            act = getattr(modelo, attr)(obs)
            return int(act)

    # callable
    if callable(modelo):
        return int(modelo(obs))

    raise RuntimeError("El objeto de modelo no expone .predict(.), .act(.), .choose_action(.) ni es callable(obs).")

def load_model_local(nombre_archivo: str) -> Any:
    ruta = os.path.join(MODELS_DIR, nombre_archivo)
    if not os.path.isfile(ruta):
        raise FileNotFoundError(f"No existe: {ruta}")
    with open(ruta, "rb") as f:
        modelo = pickle.load(f)
    modelo = _unwrap_possible_container(modelo)
    logging.info(f"Modelo cargado: {type(modelo)} — iface="
                 f"{'predict' if hasattr(modelo,'predict') else ''}"
                 f"{' act' if hasattr(modelo,'act') else ''}"
                 f"{' choose_action' if hasattr(modelo,'choose_action') else ''}"
                 f"{' callable' if callable(modelo) else ''}")
    return modelo

def maybe_pull_from_s3(nombre_archivo: str) -> Optional[str]:
    if not AUTO_PULL_S3 or not S3_BUCKET:
        return None
    try:
        import boto3
        s3 = boto3.client("s3")
        key = f"{S3_PREFIX.rstrip('/')}/{nombre_archivo}" if S3_PREFIX else nombre_archivo
        dst = os.path.join(MODELS_DIR, nombre_archivo)
        os.makedirs(MODELS_DIR, exist_ok=True)
        logging.info(f"S3 get s3://{S3_BUCKET}/{key} -> {dst}")
        s3.download_file(S3_BUCKET, key, dst)
        return dst
    except Exception as e:
        logging.warning(f"No se pudo descargar {nombre_archivo} desde S3: {e}")
        return None

def render_loop(env, action_fn):
    global last_jpeg

    obs, _ = env.reset(seed=random.randint(0, 10_000))
    done = False
    total = 0.0

    while not stop_flag.is_set() and not done:
        t0 = time.time()
        try:
            a = action_fn(obs)
        except Exception as e:
            logging.error(f"Error simulación (acción): {e}")
            break

        try:
            out = env.step(a)
            if len(out) == 5:
                obs, r, term, trunc, _ = out
                done = term or trunc
            else:
                obs, r, done, _ = out
            total += float(r)
        except Exception as e:
            logging.error(f"Error simulación (step): {e}")
            break

        try:
            frame = env.render()
            if frame is not None:
                with frame_lock:
                    last_jpeg = to_jpeg(frame)
        except Exception as e:
            logging.warning(f"Render falló: {e}")

        # Frenos: FPS + delay extra
        dt = time.time() - t0
        delay = max(0.0, (1.0 / max(1, FPS)) - dt)
        delay += max(0, STEP_DELAY_MS) / 1000.0
        time.sleep(delay)

    env.close()
    estado["recompensa_total"] = round(total, 2)
    estado["en_ejecucion"] = False
    logging.info(f"Episodio terminado. Recompensa total: {total:.2f}")

# ----------------- Rutas -----------------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/estado")
def get_estado():
    return jsonify(estado)

@app.route("/stop", methods=["POST"])
def stop():
    stop_flag.set()
    estado["en_ejecucion"] = False
    return jsonify({"ok": True})

# ------ Humano ------
@app.route("/humano/start", methods=["POST"])
def humano_start():
    if estado["en_ejecucion"]:
        return jsonify({"ok": False, "msg": "Ya hay un episodio en ejecución"}), 400
    stop_flag.clear()
    estado.update({"en_ejecucion": True, "modo": "humano", "modelo": None, "recompensa_total": 0.0})

    env = make_env()  # streaming en vivo

    def action_fn(obs):
        return decide_action_from_keys(keys_state)

    th = threading.Thread(target=render_loop, args=(env, action_fn), daemon=True)
    th.start()
    return jsonify({"ok": True})

@app.route("/humano/keys", methods=["POST"])
def humano_keys():
    data = request.get_json(force=True, silent=True) or {}
    for k in ("left", "main", "right"):
        if k in data:
            keys_state[k] = bool(data[k])
    return jsonify({"ok": True})

# ------ Demo (aleatoria) ------
@app.route("/demo", methods=["POST"])
def demo():
    if estado["en_ejecucion"]:
        return jsonify({"ok": False, "msg": "Ya hay un episodio en ejecución"}), 400
    stop_flag.clear()
    estado.update({"en_ejecucion": True, "modo": "demo", "modelo": None, "recompensa_total": 0.0})

    env = make_env()
    def action_fn(obs):
        return env.action_space.sample()
    th = threading.Thread(target=render_loop, args=(env, action_fn), daemon=True)
    th.start()
    return jsonify({"ok": True})

# ------ Modelo (.pkl) ------
@app.route("/modelo", methods=["POST"])
def modelo():
    if estado["en_ejecucion"]:
        return jsonify({"ok": False, "msg": "Ya hay un episodio en ejecución"}), 400
    data = request.get_json(force=True, silent=True) or {}
    nombre = (data.get("modelo_id") or "").strip()
    if not nombre:
        return jsonify({"ok": False, "msg": "Falta modelo_id"}), 400

    mapping = {
        "Qlearning1k":  "trained_agent_Qlearning1k.pkl",
        "Qlearning19k": "trained_agent_Qlearning19k.pkl",
        "Sarsa19k":     "trained_agent_sarsa19k.pkl",
    }
    archivo = mapping.get(nombre, nombre)

    if not os.path.isfile(os.path.join(MODELS_DIR, archivo)):
        maybe_pull_from_s3(archivo)

    try:
        if archivo in model_cache:
            modelo_obj = model_cache[archivo]
        else:
            modelo_obj = load_model_local(archivo)
            model_cache[archivo] = modelo_obj
    except Exception as e:
        logging.error(f"No pude cargar modelo {archivo}: {e}")
        return jsonify({"ok": False, "msg": f"No pude cargar {archivo}: {e}"}), 400

    stop_flag.clear()
    estado.update({"en_ejecucion": True, "modo": "modelo", "modelo": nombre, "recompensa_total": 0.0})

    env = make_env()
    env = maybe_wrap_record(env, prefix=nombre)

    def action_fn(obs):
        return decide_action_from_model(modelo_obj, obs)

    th = threading.Thread(target=render_loop, args=(env, action_fn), daemon=True)
    th.start()
    return jsonify({"ok": True})

# ------ Video MJPEG ------
@app.route("/video.mjpeg")
def video_mjpeg():
    boundary = "--frame"
    def gen():
        while True:
            with frame_lock:
                jpg = last_jpeg
            if jpg is None:
                img = Image.new("RGB", (640, 480), (0,0,0))
                buf = io.BytesIO(); img.save(buf, format="JPEG")
                jpg = buf.getvalue()
            yield (b"%s\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n" % (boundary.encode(), len(jpg))) + jpg + b"\r\n"
            time.sleep(1.0 / max(1, FPS))
    return Response(gen(), mimetype=f"multipart/x-mixed-replace; boundary=frame")

# ------ Servir MP4 grabados (opcional) ------
@app.route("/videos/")
def list_videos():
    try:
        files = [f for f in os.listdir(VIDEO_DIR) if f.endswith(".mp4")]
    except Exception:
        files = []
    html = "<h3>Videos</h3>" + "<br>".join(f'<a href="/videos/{f}">{f}</a>' for f in sorted(files))
    return html or "Sin videos"

@app.route("/videos/<path:fn>")
def get_video(fn):
    return send_from_directory(VIDEO_DIR, fn, as_attachment=False)

# ----------------- Main -----------------
if __name__ == "__main__":
    logging.info(f"Sirviendo en http://{HOST}:{PORT} (REST + MJPEG)")
    app.run(host=HOST, port=PORT, debug=False)
