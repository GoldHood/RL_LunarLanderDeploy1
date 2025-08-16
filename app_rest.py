# app_rest.py — REST + MJPEG para LunarLander (discreto)
import io, os, time, threading, pickle, logging, random
from typing import Callable, Any
from flask import Flask, render_template, Response, request, jsonify
import numpy as np
import gymnasium as gym
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

APP_FPS = float(os.getenv("APP_FPS", "24"))      # 20–30 va bien
MAX_STEPS = int(os.getenv("MAX_STEPS", "2000"))  # corte duro por si acaso

# === Flask ===========================================================
app = Flask(__name__, template_folder="templates")

# Estado compartido para el visor
estado = {
    "en_ejecucion": False,
    "modo": "inactivo",   # inactivo | humano | demo | modelo
    "modelo": None,
    "recompensa": 0.0,
}
frame_lock = threading.Lock()
frame_jpeg: bytes | None = None
hilo_actual: threading.Thread | None = None

# === Utils ===========================================================
def _np_to_jpeg(arr: np.ndarray, quality=80) -> bytes:
    img = Image.fromarray(arr)  # RGB
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    return buf.getvalue()

def _make_env() -> gym.Env:
    # ¡OJO!: v2, no v3 (v3 no existe en gymnasium 0.29.x)
    return gym.make("LunarLander-v2", render_mode="rgb_array")

class PolicyAdapter:
    """
    Adaptador para agentes heterogéneos (.pkl):
    - choose_action(obs)
    - act(obs)
    - predict(obs)   (SB3 u otros)
    - callable(obs)
    Devuelve int en {0,1,2,3}.
    """
    def __init__(self, obj: Any):
        self.obj = obj
        self._call = None

        if hasattr(obj, "choose_action") and callable(obj.choose_action):
            self._call = obj.choose_action
        elif hasattr(obj, "act") and callable(obj.act):
            self._call = obj.act
        elif hasattr(obj, "predict") and callable(obj.predict):
            def _pred(o):
                out = obj.predict(o)
                if isinstance(out, tuple):
                    return int(out[0])
                return int(out)
            self._call = _pred
        elif callable(obj):
            self._call = obj
        else:
            raise RuntimeError("El objeto del .pkl no expone choose_action/act/predict ni es callable.")

    def __call__(self, obs: np.ndarray) -> int:
        a = self._call(obs)
        try:
            a = int(a)
        except Exception:
            if isinstance(a, (list, tuple, np.ndarray)):
                a = int(a[0])
            else:
                raise
        return max(0, min(3, a))

def _cargar_modelo(nombre: str) -> PolicyAdapter:
    mapping = {
        "Qlearning1k":  "trained_agent_Qlearning1k.pkl",
        "Qlearning19k": "trained_agent_Qlearning19k.pkl",
        "Sarsa19k":     "trained_agent_sarsa19k.pkl",
    }
    if nombre not in mapping:
        raise RuntimeError(f"Modelo desconocido: {nombre}")

    candidates = [
        os.path.join("/app/models", mapping[nombre]),
        os.path.join("models",      mapping[nombre]),
    ]
    path = next((p for p in candidates if os.path.exists(p)), None)
    if not path:
        raise RuntimeError(f"No se encontró el archivo del modelo para {nombre}. Busqué: {candidates}")

    logging.info("Cargando modelo %s desde %s", nombre, path)
    with open(path, "rb") as f:
        obj = pickle.load(f)
    return PolicyAdapter(obj)

def _simular(policy: Callable[[np.ndarray], int] | None, modo: str, seed: int | None = None):
    """
    Bucle de simulación. Si policy es None => demo aleatorio.
    """
    global frame_jpeg

    env = _make_env()
    obs, _info = env.reset(seed=seed if seed is not None else random.randint(0, 10_000))

    recompensa = 0.0
    steps = 0
    try:
        estado["en_ejecucion"] = True
        estado["modo"] = modo
        estado["recompensa"] = 0.0

        terminado = False
        truncado = False

        while not (terminado or truncado):
            if modo == "humano":
                accion = 0  # si más adelante lees teclas, reemplaza por tu lectura
            else:
                accion = env.action_space.sample() if policy is None else policy(obs)

            out = env.step(int(accion))
            if len(out) == 5:
                obs, r, terminado, truncado, _ = out
            else:
                obs, r, done, _ = out
                terminado, truncado = done, False

            recompensa += float(r)
            estado["recompensa"] = float(recompensa)

            rgb = env.render()
            if rgb is not None:
                jpeg = _np_to_jpeg(rgb, quality=80)
                with frame_lock:
                    frame_jpeg = jpeg

            steps += 1
            if steps >= MAX_STEPS:
                break

            time.sleep(1.0 / APP_FPS)

    except Exception as e:
        logging.exception("Error simulación: %s", e)
    finally:
        env.close()
        estado["en_ejecucion"] = False
        estado["modo"] = "inactivo"

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/estado")
def get_estado():
    return jsonify({
        "ok": True,
        "en_ejecucion": estado["en_ejecucion"],
        "modo": estado["modo"],
        "modelo": estado["modelo"],
        "recompensa": round(float(estado["recompensa"]), 2),
    })

@app.route("/stream.mjpeg")
def stream_mjpeg():
    def gen():
        boundary = b"--frame\r\n"
        headers = b"Content-Type: image/jpeg\r\n\r\n"
        while True:
            with frame_lock:
                img = frame_jpeg
            if img:
                yield boundary + headers + img + b"\r\n"
            time.sleep(1.0 / APP_FPS)
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/demo", methods=["POST"])
def start_demo():
    global hilo_actual
    if estado["en_ejecucion"]:
        return jsonify({"ok": False, "msg": "hay una simulación en curso"}), 409
    estado["modelo"] = None
    hilo_actual = threading.Thread(target=_simular, args=(None, "demo", random.randint(0, 10_000)), daemon=True)
    hilo_actual.start()
    return jsonify({"ok": True})

@app.route("/modelo", methods=["POST"])
def start_modelo():
    global hilo_actual
    data = request.get_json(silent=True) or {}
    nombre = data.get("nombre")
    try:
        policy = _cargar_modelo(nombre)
    except Exception as e:
        logging.exception("Error cargando modelo")
        return jsonify({"ok": False, "msg": str(e)}), 500

    if estado["en_ejecucion"]:
        return jsonify({"ok": False, "msg": "hay una simulación en curso"}), 409

    estado["modelo"] = nombre
    hilo_actual = threading.Thread(target=_simular, args=(policy, "modelo", random.randint(0, 10_000)), daemon=True)
    hilo_actual.start()
    return jsonify({"ok": True})

@app.route("/stop", methods=["POST"])
def stop():
    return jsonify({"ok": True, "msg": "la simulación actual terminará sola"})

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    app.run(host="0.0.0.0", port=port, debug=False)
