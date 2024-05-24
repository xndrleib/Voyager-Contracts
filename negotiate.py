from voyager.negotiation import Negotiation, Negotiator
import json

model = "gpt-4o"  # "gpt-3.5-turbo" | "gpt-4" | "gpt-4-turbo" | "gpt-4o"

scenario = 'cleanup'  # 'swapping' | 'cleanup'

with open(f'scenarios/{scenario}.json') as f:
    data = json.load(f)
    scenario_description = data['description']
    task1 = data['tasks']['Gizmo']
    task2 = data['tasks']['Glitch']

# task1 += ' To make a fair deal you should consider hardness of obtaining of items you have in your inventory'
# task2 += ' To make a fair deal you should consider hardness of obtaining of items you have in your inventory'

# context1 = "Inventory (1/36): {'obsidian': 26}"
# context2 = "Inventory (2/36): {'diamond': 2, 'book': 12}"
context1 = "Inventory (1/36): {'obsidian': 32}"
context2 = "Inventory (1/36): {'obsidian': 26}"

agent1 = Negotiator(name="Gizmo", task=task1, other_name="Glitch", other_task=task2,
                    scenario=scenario_description, model=model, context=context1)
agent2 = Negotiator(name="Glitch", task=task2, other_name="Gizmo", other_task=task1,
                    scenario=scenario_description, model=model, context=context2)

conversation = Negotiation(agent1, agent2, max_turns=10)
conversation.simulate()

print('Agent 1 Log\n')
print(agent1.prepare_conversation_string())

print('\n\nAgent 2 Log\n')
print(agent2.prepare_conversation_string())
