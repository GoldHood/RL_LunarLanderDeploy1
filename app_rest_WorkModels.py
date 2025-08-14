# app_rest.py
# ------------------------------------------------------------
# Flask con API REST + streaming MJPEG para ver LunarLander-v2
# sin dependencias de Socket.IO ni eventlet.
#
# Endpoints:
#  - GET  /                -> index.html
#  - GET  /estado          -> JSON con estado del episodio
#  - POST /demo            -> inicia episodio con heurística simple
#  - POST /modelo          -> inicia episodio con un modelo SB3 (si existe)
#  - POST /stop            -> solicita parar el episodio en curso
#  - GET  /stream          -> stream MJPEG con los frames
#
# Requisitos: Flask, gymnasium[box2d], box2d-py, numpy, pillow
# (Opcional para modelos): stable-baselines3, torch
# ------------------------------------------------------------

import os
import io
import time
import threading
import logging
from typing import Optional, Dict

import numpy as np
from PIL import Image
from flask import Flask, render_template, request, jsonify, Response

import gymnasium as gym

# ---------------- Config ----------------
app = Flask(__name__, template_folder="templates")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

MODELOS_ESPERADOS = ["modelo_1", "modelo_2", "modelo_3", "modelo_4"]

# Estado global mínimo (protegido por lock)
estado: Dict = {
    "en_ejecucion": False,
    "modo": None,             # "demo" | "modelo"
    "modelo_id": None,
    "recompensa": 0.0,
    "pasos": 0,
}
estado_lock = threading.Lock()

# Control del episodio en background
hilo_simulacion: Optional[threading.Thread] = None
stop_event = threading.Event()

# Último frame JPEG listo para stream
ultimo_jpeg: Optional[bytes] = None
frame_lock = threading.Lock()

# Cache de modelos SB3 (opcional)
_modelos_cache = {}

# ---------------- Utilidades ----------------
def _to_jpeg(frame_rgb: np.ndarray, calidad: int = 80) -> bytes:
    """Convierte ndarray (H,W,3) uint8 a JPEG bytes."""
    if frame_rgb.dtype != np.uint8:
        frame_rgb = np.clip(frame_rgb, 0, 255).astype(np.uint8)
    im = Image.fromarray(frame_rgb)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=calidad)
    return buf.getvalue()

def _politica_demo(obs: np.ndarray) -> int:
    """Heurística simple para volar más o menos estable."""
    x, y, vx, vy, theta, vtheta, lc, rc = obs
    if vy < -0.4:
        return 2  # motor principal
    if theta > 0.1:
        return 1  # corrige a la izquierda
    if theta < -0.1:
        return 3  # corrige a la derecha
    return 0      # no-op

def _cargar_modelo(nombre: str):
    """Carga perezosamente un modelo SB3 si existe modelo_X.zip."""
    if nombre in _modelos_cache:
        return _modelos_cache[nombre]
    ruta = f"{nombre}.zip"
    if not os.path.exists(ruta):
        return None
    from stable_baselines3 import DQN  # ajusta aquí si tus modelos son PPO/A2C/…
    modelo = DQN.load(ruta)
    _modelos_cache[nombre] = modelo
    logging.info(f"✅ Cargado {ruta}")
    return modelo

def _actualizar_estado(**kwargs):
    with estado_lock:
        estado.update(kwargs)

def _reset_estado():
    _actualizar_estado(en_ejecucion=False, modo=None, modelo_id=None, recompensa=0.0, pasos=0)

# ---------------- Simulación en background ----------------
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
                logging.warning("Modelo no disponible; se aborta simulación.")
                return

        _actualizar_estado(en_ejecucion=True, modo=modo, modelo_id=modelo_id, recompensa=0.0, pasos=0)

        # Primer frame
        frame = env.render()
        with frame_lock:
            ultimo_jpeg = _to_jpeg(frame)

        dt = 1.0 / max(fps, 1.0)

        while not (terminado or truncado) and not stop_event.is_set():
            if modelo is None:
                accion = _politica_demo(obs)
            else:
                accion, _ = modelo.predict(obs, deterministic=True)
                accion = int(accion)

            obs, r, terminado, truncado, _ = env.step(accion)

            # Frame actual
            frame = env.render()
            with frame_lock:
                ultimo_jpeg = _to_jpeg(frame)

            # Estado
            with estado_lock:
                estado["recompensa"] += float(r)
                estado["pasos"] += 1

            time.sleep(dt)

    except Exception as e:
        logging.exception(f"Error en simulación: {e}")
    finally:
        if env is not None:
            env.close()
        stop_event.clear()
        _reset_estado()

# ---------------- Stream MJPEG ----------------
def _generador_mjpeg():
    boundary = b"--frame"
    while True:
        # Tomamos el último frame disponible
        with frame_lock:
            jpg = ultimo_jpeg
        if jpg is None:
            time.sleep(0.03)
            continue

        yield boundary + b"\r\n"
        yield b"Content-Type: image/jpeg\r\n"
        yield f"Content-Length: {len(jpg)}\r\n\r\n".encode("ascii")
        yield jpg + b"\r\n"

        # Ritmo del stream (el cliente mostrará el último frame)
        time.sleep(0.02)  # ~50 “intentos” por segundo

# ---------------- Rutas REST ----------------
@app.route("/")
def index():
    return render_template("index.html", esperados=MODELOS_ESPERADOS)

@app.get("/estado")
def get_estado():
    with estado_lock:
        return jsonify(estado)

@app.post("/demo")
def post_demo():
    global hilo_simulacion
    if estado["en_ejecucion"]:
        return jsonify({"ok": False, "msg": "Ya hay un episodio en ejecución"}), 409
    stop_event.clear()
    hilo_simulacion = threading.Thread(
        target=_simular, kwargs={"modo": "demo", "modelo_id": None, "seed": int(np.random.randint(0, 10000))},
        daemon=True
    )
    hilo_simulacion.start()
    return jsonify({"ok": True})

@app.post("/modelo")
def post_modelo():
    global hilo_simulacion
    data = request.get_json(silent=True) or {}
    modelo_id = data.get("modelo_id")
    if not modelo_id:
        return jsonify({"ok": False, "msg": "Falta 'modelo_id'"}), 400
    if estado["en_ejecucion"]:
        return jsonify({"ok": False, "msg": "Ya hay un episodio en ejecución"}), 409
    # Verificación rápida: existe el archivo?
    if not os.path.exists(f"{modelo_id}.zip"):
        return jsonify({"ok": False, "msg": f"No se encontró {modelo_id}.zip"}), 404

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

@app.get("/stream")
def stream():
    return Response(_generador_mjpeg(), mimetype="multipart/x-mixed-replace; boundary=frame")

# ---------------- Main ----------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5055))  # usa 5055 por defecto
    app.run(host="127.0.0.1", port=port, debug=False, threaded=True)
