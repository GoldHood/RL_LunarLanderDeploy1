import os
import pickle
import cloudpickle as cp
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from flask import Flask, render_template, send_file, abort, make_response
import time, uuid, glob, gc, shutil

# ---------------- TileCoder (como en el notebook) ----------------
class TileCoder:
    def __init__(self, num_tilings, tiles_per_dim, state_bounds):
        self.num_tilings   = int(num_tilings)
        self.tiles_per_dim = np.array(tiles_per_dim, dtype=int)
        self.state_bounds  = np.array(state_bounds, dtype=float)
        self.tile_widths   = (self.state_bounds[:,1] - self.state_bounds[:,0]) / (self.tiles_per_dim - 1)
        self.offsets       = (np.arange(self.num_tilings) * -1.0/self.num_tilings)[...,None] * self.tile_widths
        self._mults        = np.array([np.prod(self.tiles_per_dim[:d]) for d in range(len(self.tiles_per_dim))], dtype=int)

    def get_active_tiles(self, state):
        st = np.clip(np.asarray(state, dtype=np.float32), self.state_bounds[:,0], self.state_bounds[:,1])
        scaled = (st - self.state_bounds[:,0]) / self.tile_widths
        active = []
        base_stride = int(np.prod(self.tiles_per_dim))
        for i in range(self.num_tilings):
            base = i * base_stride
            coords = np.floor(scaled + self.offsets[i]).astype(int)
            idx = base + int(np.sum(coords * self._mults))
            active.append(idx)
        return active

# ---------------- Loader que devuelve una policy(obs)->action ----------------
def load_policy(model_path):
    with open(model_path, "rb") as f:
        try:
            obj = cp.load(f)
        except Exception:
            f.seek(0)
            obj = pickle.load(f)

    if callable(obj):
        def policy(obs):
            return int(obj(np.asarray(obs, dtype=np.float32)))
        return policy, "callable"

    if isinstance(obj, dict) and (("q_weights" in obj) or ("Q_weights" in obj)) and ("tile_coder" in obj):
        weights = obj.get("q_weights") or obj.get("Q_weights")
        tc_cfg  = obj["tile_coder"]
        if isinstance(tc_cfg, dict) and {"num_tilings","tiles_per_dim","state_bounds"}.issubset(tc_cfg.keys()):
            tc = TileCoder(tc_cfg["num_tilings"], np.array(tc_cfg["tiles_per_dim"], dtype=int), np.array(tc_cfg["state_bounds"], dtype=float))
            if "offsets" in tc_cfg:
                tc.offsets = np.array(tc_cfg["offsets"], dtype=float)
        else:
            raise RuntimeError("tile_coder no tiene los campos esperados")

        fixed_w = {}
        for k, v in (weights.items() if hasattr(weights, "items") else []):
            if isinstance(k, tuple):
                tile, a = int(k[0]), int(k[1])
            else:
                try:
                    tile, a = eval(k)
                    tile, a = int(tile), int(a)
                except Exception:
                    continue
            fixed_w[(tile, a)] = float(v)

        def policy(obs, _tc=tc, _w=fixed_w):
            active = _tc.get_active_tiles(obs)
            q = [0.0, 0.0, 0.0, 0.0]  # LunarLander: 4 acciones
            for a in range(4):
                s = 0.0
                for t in active:
                    s += _w.get((t, a), 0.0)
                q[a] = s
            return int(np.argmax(q))
        return policy, "tile_q_dict"

    for name in ("choose_action", "act", "predict"):
        if hasattr(obj, name):
            def policy(obs, _obj=obj, _name=name):
                return int(getattr(_obj, _name)(np.asarray(obs, dtype=np.float32)))
            return policy, f"method:{name}"

    raise RuntimeError("Formato de modelo no reconocido")

# ---------------- Flask ----------------
app = Flask(__name__, template_folder="templates")
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0  # no cache en send_file

MODELS_DIR = "models"
VIDEO_DIR  = "videos"
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(VIDEO_DIR,  exist_ok=True)

def run_episode(env_id, model_path):
    print(f"[INFO] Cargando modelo: {model_path}")
    policy, kind = load_policy(model_path)
    print(f"[INFO] Tipo detectado → {kind}")

    # Carpeta única por corrida para evitar locks/overwrites y cache
    run_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
    video_folder = os.path.join(VIDEO_DIR, run_id)
    os.makedirs(video_folder, exist_ok=True)

    env = gym.make(env_id, render_mode="rgb_array")
    env = RecordVideo(env, video_folder=video_folder, episode_trigger=lambda ep: ep == 0)

    obs, _ = env.reset()
    done = truncated = False
    while not (done or truncated):
        action = policy(obs)
        obs, reward, done, truncated, _ = env.step(action)

    env.close()
    del env
    gc.collect()

    mp4s = sorted(glob.glob(os.path.join(video_folder, "*.mp4")), key=os.path.getmtime)
    if not mp4s:
        raise RuntimeError("No se generó ningún .mp4 en la corrida")
    return mp4s[-1]  # último video de esta corrida

@app.route("/")
def index():
    return render_template("index_simple.html")

@app.route("/run/<model_name>")
def run_model(model_name):
    model_path = os.path.join(MODELS_DIR, model_name)
    if not os.path.exists(model_path):
        abort(404, f"Modelo no encontrado: {model_name}")

    mp4_path = run_episode("LunarLander-v3", model_path)

    # Enviar el archivo deshabilitando cache del navegador
    resp = make_response(send_file(mp4_path, mimetype="video/mp4"))
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    resp.headers["Pragma"] = "no-cache"
    resp.headers["Expires"] = "0"
    return resp

if __name__ == "__main__":
    # threaded=True evita que una solicitud bloquee a la siguiente
    app.run(host="0.0.0.0", port=5000, threaded=True)
