import argparse
from api_keys import openai_api_key
from voyager import MultiAgentVoyager

# Argument parser
parser = argparse.ArgumentParser(description='Running Voyager with different sets of parameters.')
parser.add_argument('--port', type=int, required=True, help='MC port number')
parser.add_argument('--server_port', type=int, default=3000, help='Server port number (default: 3000)')
args = parser.parse_args()

azure_login = None
mc_port = args.port
server_port = args.server_port

model = "gpt-4-turbo"  # "gpt-4-0613" | "gpt-4-turbo" | gpt-3.5-turbo

options = {
    'azure_login': azure_login,
    'mc_port': mc_port,
    'openai_api_key': openai_api_key,
    'resume': False,
    'env_wait_ticks': 80,
    'action_agent_task_max_retries': 50,
    'action_agent_show_chat_log': True,
    'action_agent_temperature': 0.1
}

multi_agent = MultiAgentVoyager(options=options)

# save_options = {
#     'file_name' : "cleanup3.json",
#     'scenario_block_types' : ["wet_sponge", "sweet_berry_bush"],
#     'center_position' : {'x':342, 'y': 119, 'z': 107},
#     'remove_blocks' : False
# }

save_options = {
    'file_name': "temp.json",
    'scenario_block_types': ["slime_block", "red_mushroom_block", "mushroom_stem"],
    'center_position': {"x": 333, "y": 119, "z": 123},
    'remove_blocks': True
}
multi_agent.save_scenario(save_options)
# multi_agent.load_scenario()
