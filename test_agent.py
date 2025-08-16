import os
import pickle
import gymnasium as gym
from agent_utils import select_action

# === Buscar todos los .pkl en la carpeta models ===
MODELS_DIR = "models"
pkl_files = [f for f in os.listdir(MODELS_DIR) if f.endswith(".pkl")]

if not pkl_files:
    raise FileNotFoundError("❌ No se encontraron archivos .pkl en la carpeta 'models/'")

print("📂 Modelos disponibles:")
for i, f in enumerate(pkl_files):
    print(f"{i+1}. {f}")

# Elegir modelo
choice = int(input("\n👉 Selecciona el modelo a cargar (número): ")) - 1
model_path = os.path.join(MODELS_DIR, pkl_files[choice])

print(f"\n✅ Cargando modelo: {model_path}")

# === Cargar modelo ===
with open(model_path, "rb") as f:
    agent = pickle.load(f)

# === Crear entorno LunarLander ===
env = gym.make("LunarLander-v3", render_mode=None)

state, _ = env.reset()
done = False
total_reward = 0

print("\n🚀 Ejecutando un episodio...\n")

while not done:
    action = select_action(agent, state, env.action_space)
    state, reward, terminated, truncated, _ = env.step(action)
    total_reward += reward
    done = terminated or truncated

print(f"\n🏁 Episodio terminado. Recompensa total: {total_reward:.2f}")
env.close()

