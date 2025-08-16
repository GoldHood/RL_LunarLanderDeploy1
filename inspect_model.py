import pickle

model_path = "models/trained_agent_Qlearning19k.pkl"

with open(model_path, "rb") as f:
    agent = pickle.load(f)

print("=== Tipo de agente cargado ===")
print(type(agent))

print("=== Atributos disponibles ===")
print(dir(agent))
