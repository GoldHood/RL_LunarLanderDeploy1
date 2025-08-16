# inspect_all_models.py
import os, pickle

MODELS_DIR = "models"   # ajusta si tus .pkl están en otra carpeta

for fname in os.listdir(MODELS_DIR):
    if fname.endswith(".pkl"):
        fpath = os.path.join(MODELS_DIR, fname)
        print("="*50)
        print(f"Archivo: {fname}")
        try:
            with open(fpath, "rb") as f:
                agent = pickle.load(f)
            print("Tipo:", type(agent))
            print("Atributos:", dir(agent)[:20])  # primeros 20
        except Exception as e:
            print("❌ Error cargando:", e)
