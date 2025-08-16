# agent_utils.py
import numpy as np

def select_action(agent, state, action_space):
    # Caso 1: si el agente es una función pickled
    if callable(agent):
        try:
            return int(agent(state))
        except Exception as e:
            print("[ERROR] función del agente falló:", e)

    # Caso 2: si el agente es objeto con métodos conocidos
    if hasattr(agent, "choose_action"):
        return int(agent.choose_action(state))
    if hasattr(agent, "act"):
        return int(agent.act(state))
    if hasattr(agent, "predict"):
        return int(agent.predict(state))

    # fallback: random
    return action_space.sample()
