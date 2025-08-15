# app_rest.py — REST + MJPEG + control humano + marcador final + soporte S3
import os, io, time, threading, logging, importlib
from typing import Optional, Dict

import numpy as np
from PIL import Image
from flask import Flask, render_template, request, jsonify, Response
import gymnasium as gym

# ---------- Config ----------
app = Flask(__name__, template_folder="templates")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

MODELOS_ESPERADOS = ["modelo_1", "modelo_2", "modelo_3", "modelo_4"]

# Directorio local donde se buscan/guardan modelos
MODELOS_DIR = os.environ.get("MODELS_DIR", "/app/models")
os.makedirs(MODELOS_DIR, exist_ok=True)

# S3 (opcional)
S3_BUCKET = os.environ.get("S3_BUCKET", "").strip() or None
S3_PREFIX = os.environ.get("S3_PREFIX", "").strip()
AUTO_PULL_S3 = os.environ.get("AUTO_PULL_S3", "1") not in ("0", "false", "False")

# ---------- Estado compartido ----------
estado: Dict = {
    "en_ejecucion": False,
    "modo": None,
    "modelo_id": None,
    "recompensa": 0.0,
    "pasos": 0,
    "ultimo_puntaje": None,
}
estado_lock = threading.Lock()

hilo_simulacion: Optional[threading.Thread] = None
stop_event = threading.Event()

ultimo_jpeg: Optional[bytes] = None
frame_lock = threading.Lock()

_modelos_cache = {}  # nombre -> wrapper
control = {"enabled": False, "keys": {"up": False, "left": False, "right": False}}
control_lock = threading.Lock()

# ---------- Utils ----------
def _to_jpeg(frame_rgb: np.ndarray, calidad: int = 80) -> bytes:
    if frame_rgb.dtype != np.uint8:
        frame_rgb = np.clip(frame_rgb, 0, 255).astype(np.uint8)
    im = Image.fromarray(frame_rgb)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=calidad)
    return buf.getvalue()

def _politica_demo(obs: np.ndarray) -> int:
    x, y, vx, vy, theta, vtheta, lc, rc = obs
    if vy < -0.4: return 2
    if theta > 0.1: return 1
    if theta < -0.1: return 3
    return 0

def _accion_desde_teclas(keys: Dict[str, bool]) -> int:
    if keys.get("up"): return 2
    if keys.get("left"): return 1
    if keys.get("right"): return 3
    return 0

def _local_path(modelo_id: str, ext: str) -> str:
    return os.path.join(MODELOS_DIR, f"{modelo_id}{ext}")

# ---------- Carga/descarga de modelos ----------
def _s3_try_download(modelo_id: str) -> Optional[str]:
    """
    Si S3 está configurado, intenta descargar .zip o .pkl a MODELOS_DIR.
    Devuelve la ruta local si lo logró, si no None.
    """
    if not (S3_BUCKET and AUTO_PULL_S3):
        return None
    try:
        import boto3
    except Exception:
        logging.warning("boto3 no disponible; omitiendo S3.")
        return None

    s3 = boto3.client("s3")
    for ext in (".zip", ".pkl"):
        key = f"{S3_PREFIX}{modelo_id}{ext}"
        dest = _local_path(modelo_id, ext)
        try:
            logging.info(f"Intentando S3 get s3://{S3_BUCKET}/{key} -> {dest}")
            os.makedirs(MODELOS_DIR, exist_ok=True)
            s3.download_file(S3_BUCKET, key, dest)
            logging.info(f"Descargado {key}")
            return dest
        except Exception as e:
            logging.info(f"No se encontró {key} en S3 ({e.__class__.__name__})")
    return None

class _SB3Wrapper:
    def __init__(self, algo):
        self.algo = algo
    def predict(self, obs):
        a, _ = self.algo.predict(obs, deterministic=True)
        return int(a)

class _CustomWrapper:
    def __init__(self, obj):
        self.obj = obj
    def predict(self, obs):
        # Intentamos 'predict' o 'act' del objeto
        if hasattr(self.obj, "predict"):
            return int(self.obj.predict(obs))
        if hasattr(self.obj, "act"):
            return int(self.obj.act(obs))
        raise RuntimeError("El objeto .pkl no expone predict(obs) ni act(obs).")

def _cargar_modelo(modelo_id: str):
    """
    Carga desde cache o disco (y si falta, intenta S3).
    - .zip: intenta varios algoritmos SB3
    - .pkl: objeto python con predict/act
    """
    if modelo_id in _modelos_cache:
        return _modelos_cache[modelo_id]

    # 1) ¿Existe local?
    ruta = None
    for ext in (".zip", ".pkl"):
        p = _local_path(modelo_id, ext)
        if os.path.exists(p):
            ruta = p
            break

    # 2) Si no existe, intentamos S3
    if ruta is None:
        ruta = _s3_try_download(modelo_id)

    if ruta is None:
        return None

    # 3) Cargar según extensión
    if ruta.endswith(".zip"):
        # Probar varios algos SB3
        try:
            import stable_baselines3 as sb3
            for cls_name in ("DQN", "PPO", "A2C", "SAC", "TD3"):
                if hasattr(sb3, cls_name):
                    try:
                        algo = getattr(sb3, cls_name).load(ruta)
                        wrapper = _SB3Wrapper(algo)
                        _modelos_cache[modelo_id] = wrapper
                        logging.info(f"✅ Cargado SB3 ({cls_name}) desde {ruta}")
                        return wrapper
                    except Exception:
                        continue
            raise RuntimeError("No se pudo cargar el .zip con los algoritmos conocidos de SB3.")
        except Exception as e:
            logging.exception(f"Error cargando modelo SB3: {e}")
            return None

    if ruta.endswith(".pkl"):
        import pickle
        with open(ruta, "rb") as f:
            obj = pickle.load(f)
        wrapper = _CustomWrapper(obj)
        _modelos_cache[modelo_id] = wrapper
        logging.info(f"✅ Cargado custom .pkl desde {ruta}")
        return wrapper

    return None

def _actualizar_estado(**kwargs):
    with estado_lock:
        estado.update(kwargs)

def _reset_estado():
    _actualizar_estado(en_ejecucion=False, modo=None, modelo_id=None, recompensa=0.0, pasos=0)

# ---------- Simulación ----------
def _simular(modo: str, modelo_id: Optional[str] = None, seed: Optional[int] = None, fps: float = 60.0):
    global ultimo_jpeg
    env = None
    try:
        env = gym.make("LunarLander-v2", render_mode="rgb_array")
        obs, _ = env.reset(seed=seed)
        terminado = truncado = False

        modelo = None
        if modo == "modelo":
            modelo = _cargar_modelo(modelo_id) if modelo_id else None
            if modelo is None:
                logging.warning("Modelo no disponible; abortando simulación.")
                return

        _actualizar_estado(en_ejecucion=True, modo=modo, modelo_id=modelo_id, recompensa=0.0, pasos=0)

        frame = env.render()
        with frame_lock:
            ultimo_jpeg = _to_jpeg(frame)

        dt = 1.0 / max(fps, 1.0)

        while not (terminado or truncado) and not stop_event.is_set():
            if modo == "humano":
                with control_lock:
                    accion = _accion_desde_teclas(control["keys"])
            elif modelo is None:
                accion = _politica_demo(obs)
            else:
                accion = modelo.predict(obs)

            obs, r, terminado, truncado, _ = env.step(accion)

            frame = env.render()
            with frame_lock:
                ultimo_jpeg = _to_jpeg(frame)

            with estado_lock:
                estado["recompensa"] += float(r)
                estado["pasos"] += 1

            time.sleep(dt)

    except Exception as e:
        logging.exception(f"Error en simulación: {e}")
    finally:
        if env is not None:
            env.close()
        with estado_lock:
            estado["ultimo_puntaje"] = estado["recompensa"]
        stop_event.clear()
        with control_lock:
            control["enabled"] = False
        _reset_estado()

# ---------- Stream MJPEG ----------
def _generador_mjpeg():
    boundary = b"--frame"
    while True:
        with frame_lock:
            jpg = ultimo_jpeg
        if jpg is None:
            time.sleep(0.02); continue
        yield boundary + b"\r\n"
        yield b"Content-Type: image/jpeg\r\n"
        yield f"Content-Length: {len(jpg)}\r\n\r\n".encode("ascii")
        yield jpg + b"\r\n"
        time.sleep(0.02)

# ---------- Rutas ----------
@app.route("/")
def index():
    return render_template("index.html", esperados=MODELOS_ESPERADOS)

@app.get("/estado")
def get_estado():
    with estado_lock:
        return jsonify(estado)

@app.get("/stream")
def stream():
    return Response(_generador_mjpeg(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.post("/demo")
def post_demo():
    global hilo_simulacion
    if estado["en_ejecucion"]:
        return jsonify({"ok": False, "msg": "Ya hay un episodio en ejecución"}), 409
    stop_event.clear()
    hilo_simulacion = threading.Thread(
        target=_simular, kwargs={"modo": "demo", "seed": int(np.random.randint(0, 10000))}, daemon=True
    )
    hilo_simulacion.start()
    return jsonify({"ok": True})

@app.post("/modelo")
def post_modelo():
    """ body: {"modelo_id": "modelo_1"}  (opcional: {"s3_key": "ruta/en/s3.pkl"}) """
    global hilo_simulacion
    data = request.get_json(silent=True) or {}
    modelo_id = data.get("modelo_id")
    s3_key = data.get("s3_key")

    if estado["en_ejecucion"]:
        return jsonify({"ok": False, "msg": "Ya hay un episodio en ejecución"}), 409

    # Permite forzar descarga por s3_key explícito
    if s3_key and S3_BUCKET:
        try:
            import boto3, pathlib
            s3 = boto3.client("s3")
            base = os.path.basename(s3_key)
            modelo_id = os.path.splitext(base)[0]
            destino = os.path.join(MODELOS_DIR, base)
            s3.download_file(S3_BUCKET, f"{S3_PREFIX}{s3_key}", destino)
            logging.info(f"Descargado {s3_key} como {destino}")
        except Exception as e:
            return jsonify({"ok": False, "msg": f"Fallo S3: {e}"}), 400

    if not modelo_id:
        return jsonify({"ok": False, "msg": "Falta 'modelo_id'"}), 400

    # Validar que exista o se pueda bajar
    loc = None
    for ext in (".zip", ".pkl"):
        p = _local_path(modelo_id, ext)
        if os.path.exists(p):
            loc = p; break
    if loc is None:
        loc = _s3_try_download(modelo_id)
    if loc is None:
        return jsonify({"ok": False, "msg": f"No se encontró {modelo_id} (.zip/.pkl) local ni en S3"}), 404

    stop_event.clear()
    hilo_simulacion = threading.Thread(
        target=_simular, kwargs={"modo": "modelo", "modelo_id": modelo_id, "seed": int(np.random.randint(0, 10000))},
        daemon=True
    )
    hilo_simulacion.start()
    return jsonify({"ok": True})

@app.post("/stop")
def post_stop():
    if not estado["en_ejecucion"]:
        return jsonify({"ok": True, "msg": "No había episodio en curso"})
    stop_event.set()
    return jsonify({"ok": True})

# --- Humano ---
@app.post("/humano/start")
def humano_start():
    global hilo_simulacion
    if estado["en_ejecucion"]:
        return jsonify({"ok": False, "msg": "Ya hay un episodio en ejecución"}), 409
    with control_lock:
        control["enabled"] = True
        control["keys"] = {"up": False, "left": False, "right": False}
    stop_event.clear()
    hilo_simulacion = threading.Thread(
        target=_simular, kwargs={"modo": "humano", "seed": int(np.random.randint(0, 10000))}, daemon=True
    )
    hilo_simulacion.start()
    return jsonify({"ok": True})

@app.post("/humano/keys")
def humano_keys():
    data = request.get_json(silent=True) or {}
    with control_lock:
        for k in ("up", "left", "right"):
            if k in data: control["keys"][k] = bool(data[k])
    return jsonify({"ok": True, "keys": control["keys"]})

@app.post("/humano/stop")
def humano_stop():
    stop_event.set()
    return jsonify({"ok": True})

# --- Gestión de modelos ---
@app.post("/models/refresh")
def models_refresh():
    """Intenta bajar todos los MODELOS_ESPERADOS desde S3 si faltan localmente."""
    if not S3_BUCKET: return jsonify({"ok": False, "msg": "S3 no configurado"}), 400
    bajados = []
    for m in MODELOS_ESPERADOS:
        have = any(os.path.exists(_local_path(m, ext)) for ext in (".zip", ".pkl"))
        if not have:
            path = _s3_try_download(m)
            if path: bajados.append(os.path.basename(path))
    return jsonify({"ok": True, "descargados": bajados})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    logging.info(f"Sirviendo en http://{host}:{port} (REST + MJPEG + humano + S3)")
    app.run(host=host, port=port, debug=False, threaded=True)

