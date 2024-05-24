import logging
import re
import time

import voyager.utils as U
from javascript import require
from langchain.chat_models import ChatOpenAI
from langchain.prompts import SystemMessagePromptTemplate
from langchain.schema import AIMessage, HumanMessage, SystemMessage

from voyager.prompts import load_prompt
from voyager.control_primitives_context import load_control_primitives_context


class ActionAgent:
    def __init__(
            self,
            model_name="gpt-3.5-turbo",
            temperature=0,
            request_timout=120,
            episode_timeout=120,
            ckpt_dir="ckpt",
            resume=False,
            chat_log=True,
            execution_error=True,
            logger=None,
            username="Voyager",
    ):
        self.ckpt_dir = ckpt_dir
        self.chat_log = chat_log
        self.execution_error = execution_error
        self.episode_timeout = episode_timeout
        self.logger = logger
        self.username = username
        U.f_mkdir(f"{ckpt_dir}/action")
        if resume:
            self.logger(f"\033[32mLoading Action Agent from {ckpt_dir}/action\033[0m")
            self.chest_memory = U.load_json(f"{ckpt_dir}/action/chest_memory.json")
        else:
            self.chest_memory = {}
        self.llm = ChatOpenAI(
            model_name=model_name,
            temperature=temperature,
            request_timeout=request_timout,
        )

    def update_chest_memory(self, chests):
        for position, chest in chests.items():
            if position in self.chest_memory:
                if isinstance(chest, dict):
                    self.chest_memory[position] = chest
                if chest == "Invalid":
                    self.logger(
                        f"\033[32mAction Agent removing chest {position}: {chest}\033[0m"
                    )
                    self.chest_memory.pop(position)
            else:
                if chest != "Invalid":
                    self.logger(f"\033[32mAction Agent saving chest {position}: {chest}\033[0m")
                    self.chest_memory[position] = chest
        U.dump_json(self.chest_memory, f"{self.ckpt_dir}/action/chest_memory.json")

    def render_chest_observation(self):
        chests = []
        for chest_position, chest in self.chest_memory.items():
            if isinstance(chest, dict) and len(chest) > 0:
                chests.append(f"{chest_position}: {chest}")
        for chest_position, chest in self.chest_memory.items():
            if isinstance(chest, dict) and len(chest) == 0:
                chests.append(f"{chest_position}: Empty")
        for chest_position, chest in self.chest_memory.items():
            if isinstance(chest, str):
                assert chest == "Unknown"
                chests.append(f"{chest_position}: Unknown items inside")
        assert len(chests) == len(self.chest_memory)
        if chests:
            chests = "\n".join(chests)
            return f"Chests:\n{chests}\n\n"
        else:
            return f"Chests: None\n\n"

    def render_system_message(self, skills=[], username=""):
        system_template = load_prompt("action_template")
        # FIXME: Hardcoded control_primitives
        base_skills = [
            # "exploreUntil",
            "mineBlock",
            # "craftItem",
            # "placeItem",
            # "multiAgent",
            # "farm",
            # "smeltItem",
            # "killMob",
        ]
        if not self.llm.model_name == "gpt-3.5-turbo":
            base_skills += [
                # "useChest",
                "mineflayer",
            ]
        programs = "\n\n".join(load_control_primitives_context(base_skills) + skills)
        response_format = load_prompt("action_response_format")
        system_message_prompt = SystemMessagePromptTemplate.from_template(
            system_template
        )
        system_message = system_message_prompt.format(
            username=self.username, programs=programs, response_format=response_format
        )
        assert isinstance(system_message, SystemMessage)
        return system_message

    def extract_event_data(self, events):
        chat_messages = []
        error_messages = []
        damage_messages = []

        biome = None
        time_of_day = None
        voxels = []
        entities = {}
        health = None
        hunger = None
        position = {}
        equipment = []
        inventory_used = 0
        inventory = []
        username = ""

        assert events[-1][0] == "observe", "Last event must be observe"

        for i, (event_type, event) in enumerate(events):
            if event_type == "onChat":
                chat_messages.append(event["onChat"])
            elif event_type == "onError":
                error_messages.append(event["onError"])
            elif event_type == "onDamage":
                damage_messages.append(event["onDamage"])
            elif event_type == "observe":
                biome = event["status"]["biome"]
                time_of_day = event["status"]["timeOfDay"]
                voxels = event["voxels"]
                entities = event["status"]["entities"]
                health = event["status"]["health"]
                hunger = event["status"]["food"]
                position = event["status"]["position"]
                equipment = event["status"]["equipment"]
                inventory_used = event["status"]["inventoryUsed"]
                inventory = event["inventory"]
                username = event["status"]["name"]
                assert i == len(events) - 1, "observe must be the last event"

        return {
            "chat_messages": chat_messages,
            "error_messages": error_messages,
            "damage_messages": damage_messages,
            "biome": biome,
            "time_of_day": time_of_day,
            "voxels": voxels,
            "entities": entities,
            "health": health,
            "hunger": hunger,
            "position": position,
            "equipment": equipment,
            "inventory_used": inventory_used,
            "inventory": inventory,
            "username": username
        }

    def create_observation_string(self, event_data, features=None, **kwargs):
        if features is None:
            features = [
                "code", "error", "chat", "biome", "time", "voxels", "entities",
                "health", "hunger", "position", "inventory", "chest_observation",
                "username", "scenario", "task", "contract", "context", "critique",
                "contract_critique"
            ]

        chat_messages = event_data["chat_messages"]
        error_messages = event_data["error_messages"]
        biome = event_data["biome"]
        time_of_day = event_data["time_of_day"]
        voxels = event_data["voxels"]
        entities = event_data["entities"]
        health = event_data["health"]
        hunger = event_data["hunger"]
        position = event_data["position"]
        inventory_used = event_data["inventory_used"]
        inventory = event_data["inventory"]
        username = event_data["username"]

        observation = ""

        if "code" in features:
            code = kwargs.get("code", "")
            if code:
                observation += f"Code from the last round:\n{code}\n\n"
            else:
                observation += f"Code from the last round: No code in the first round\n\n"

        if "error" in features and self.execution_error:
            if error_messages:
                error = "\n".join(error_messages)
                observation += f"Execution error:\n{error}\n\n"
            else:
                observation += f"Execution error: No error\n\n"

        if "chat" in features and self.chat_log:
            if chat_messages:
                chat_log = "\n".join(chat_messages)
                observation += f"Chat log: {chat_log}\n\n"
            else:
                observation += f"Chat log: None\n\n"

        if "biome" in features:
            observation += f"Biome: {biome}\n\n"

        if "time" in features:
            observation += f"Time: {time_of_day}\n\n"

        if "voxels" in features:
            if voxels:
                observation += f"Nearby blocks: {', '.join(voxels)}\n\n"
            else:
                observation += f"Nearby blocks: None\n\n"

        if "entities" in features:
            if entities:
                nearby_entities = [
                    k for k, v in sorted(entities.items(), key=lambda x: x[1])
                ]
                observation += f"Nearby entities (nearest to farthest): {', '.join(nearby_entities)}\n\n"
            else:
                observation += f"Nearby entities (nearest to farthest): None\n\n"

        if "health" in features:
            observation += f"Health: {health:.1f}/20\n\n"

        if "hunger" in features:
            observation += f"Hunger: {hunger:.1f}/20\n\n"

        if "position" in features:
            observation += f"Position: x={position['x']:.1f}, y={position['y']:.1f}, z={position['z']:.1f}\n\n"

        if "inventory" in features:
            if inventory:
                observation += f"Inventory ({inventory_used}/36): {inventory}\n\n"
            else:
                observation += f"Inventory ({inventory_used}/36): Empty\n\n"

        if "chest_observation" in features and not (
                kwargs.get("task", "") == "Place and deposit useless items into a chest"
                or kwargs.get("task", "").startswith("Deposit useless items into the chest at")
        ):
            observation += self.render_chest_observation()

        if "username" in features:
            observation += f"Username: You are {username}\n\n"

        if "scenario" in features:
            observation += f"Scenario: {kwargs.get('scenario', '')}\n\n"

        if "task" in features:
            observation += f"Task: {kwargs.get('task', '')}\n\n"

        if "contract" in features:
            observation += f"Contract: {kwargs.get('contract', '')}\n\n"

        if "context" in features:
            context = kwargs.get("context", "")
            if context:
                observation += f"Context: {context}\n\n"
            else:
                observation += f"Context: None\n\n"

        if "critique" in features:
            critique = kwargs.get("critique", "")
            if critique:
                observation += f"Task Critique: {critique}\n\n"
            else:
                observation += f"Task Critique: None\n\n"

        if "contract_critique" in features:
            contract_critique = kwargs.get("contract_critique", "")
            if contract_critique:
                observation += f"Contract Critique: {contract_critique}\n\n"
            else:
                observation += f"Contract Critique: None\n\n"

        return observation

    def render_human_message(
            self, *, events, code="", task="", contract="", scenario="", context="", critique="", contract_critique=""
    ):
        event_data = self.extract_event_data(events)

        observation = self.create_observation_string(
            event_data=event_data, features=[
                "code", "error", "chat", "biome", "time", "voxels", "entities",
                "health", "hunger", "position", "inventory", "chest_observation",
                "username", "scenario", "task", "contract", "context",
                "critique", "contract_critique"
            ], code=code, task=task, contract=contract, scenario=scenario, context=context, critique=critique,
            contract_critique=contract_critique
        )

        return HumanMessage(content=observation)

    def process_ai_message(self, message):
        assert isinstance(message, AIMessage)

        retry = 3
        while retry > 0:
            try:
                babel = require("@babel/core")
                babel_generator = require("@babel/generator").default
                code_pattern = re.compile(r"```(?:javascript|js)(.*?)```", re.DOTALL)
                code = "\n".join(code_pattern.findall(message.content))
                logging.debug(f'process_ai_message: Extracted {len(code)} javascript code symbols from the AI message. '
                              f'Parsing...')
                parsed = babel.parse(code)
                logging.debug(f'process_ai_message: Code is parsed')

                functions = []
                assert len(list(parsed.program.body)) > 0, "No functions found"
                for i, node in enumerate(parsed.program.body):
                    if node.type != "FunctionDeclaration":
                        continue
                    node_type = (
                        "AsyncFunctionDeclaration"
                        if node["async"]
                        else "FunctionDeclaration"
                    )
                    functions.append(
                        {
                            "name": node.id.name,
                            "type": node_type,
                            "body": babel_generator(node).code,
                            "params": list(node["params"]),
                        }
                    )

                # Filter functions that match the criteria
                candidate_functions = []

                for function in functions:
                    if function["type"] == "AsyncFunctionDeclaration":
                        if len(function["params"]) == 1 and function["params"][0].name == "bot":
                            candidate_functions.append(function)

                if len(candidate_functions) > 1:
                    logging.warning((f"Expected exactly one main function with a single 'bot' parameter, "
                                     f"found {len(candidate_functions)}:\n{candidate_functions}."
                                     f"Continue with the last one"))

                main_function = candidate_functions[-1]

                program_code = "\n\n".join(function["body"] for function in functions)

                exec_code = f"""
const result = await Promise.race([
    {main_function['name']}(bot),
    new Promise(resolve => setTimeout(() => resolve('Timeout reached'), {self.episode_timeout * 1000}))
]);

if (result === 'Timeout reached') {{
    bot.chat('[episode timeout]');
}}"""
                return {
                    "program_code": program_code,
                    "program_name": main_function["name"],
                    "exec_code": exec_code,
                }
            except Exception as e:
                logging.debug(f'process_ai_message: Error parsing action response (before program execution): {e}.'
                              f'The AI message content:\n{message.content}')
                retry -= 1
        return None

    def summarize_chatlog(self, events):
        def filter_item(message: str):
            craft_pattern = r"I cannot make \w+ because I need: (.*)"
            craft_pattern2 = (
                r"I cannot make \w+ because there is no crafting table nearby"
            )
            mine_pattern = r"I need at least a (.*) to mine \w+!"
            if re.match(craft_pattern, message):
                return re.match(craft_pattern, message).groups()[0]
            elif re.match(craft_pattern2, message):
                return "a nearby crafting table"
            elif re.match(mine_pattern, message):
                return re.match(mine_pattern, message).groups()[0]
            else:
                return ""

        chatlog = set()
        for event_type, event in events:
            if event_type == "onChat":
                item = filter_item(event["onChat"])
                if item:
                    chatlog.add(item)
        return "I also need " + ", ".join(chatlog) + "." if chatlog else ""
