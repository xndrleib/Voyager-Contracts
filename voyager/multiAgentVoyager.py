import copy
import threading
import time
from datetime import datetime

import requests

import voyager.utils as U
from voyager import Voyager
from voyager.negotiation import Negotiation, Negotiator

import logging

logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S',
                    filename=f'logs/program/{datetime.now().strftime("%Y-%m-%d %H%M%S")}.log',
                    encoding='utf-8', level=logging.DEBUG)


class MultiAgentVoyager:
    def __init__(self, num_agents=2, server_port=3000, usernames=("Gizmo", "Glitch"), judge_username="Judy",
                 scenario_file=None, save_dir=None, critic_mode="auto", contract_mode="auto", contract=None,
                 continuous=True, episode_timeout=120, num_episodes=3, negotiator_model_name="gpt-4",
                 negotiator_temperature=0.7, skinurls=None, options=None):

        if skinurls is None:
            skinurls = ["https://images2.imgbox.com/60/3d/2bJnlM8U_o.png",  # player 1 skin
                        "https://images2.imgbox.com/a7/6c/hZRGGRAS_o.png"]  # player 2 skin
        if options is None:
            options = {}

        self.scenario_file = scenario_file
        self.scenario_description = None
        self.scenario_code = None
        self.critic_mode = critic_mode
        self.continuous = continuous
        self.contract_mode = contract_mode
        self.contract = contract
        self.agents = []
        self.judge = None
        self.usernames = usernames
        self.judge_username = judge_username
        self.num_episodes = num_episodes
        self.negotiator_model_name = negotiator_model_name
        self.negotiator_temperature = negotiator_temperature
        self.skinurls = skinurls
        self.chest_memory = {}
        self.episode = 0
        self.load_from_save = False
        self.reward_item_names = None

        assert critic_mode in ["auto", "manual"]
        assert contract_mode in ["auto", "manual"]
        if self.continuous:
            assert isinstance(self.num_episodes, int) and self.num_episodes > 0

        if self.contract_mode == "manual":
            if contract is None:
                raise ValueError("Contract mode is manual but no contract was provided")
            if not isinstance(contract, str):
                raise ValueError("Contract must be a string")
            self.contract = contract

        if num_agents != 2:
            raise ValueError("Only 2 agents are supported at this time")

        # load game save directory if it exists
        if save_dir is not None and U.f_not_empty(save_dir):
            print("Provided save directory exists. Loading game...")
            self.save_dir = save_dir

            # recover contract
            try:
                with open(f"{self.save_dir}/contract.txt", 'r') as contract_file:
                    self.contract = contract_file.read()
                if contract_mode == "auto":
                    print("Warning: contract mode is auto but contract was found in save directory. Overwriting with "
                          "saved contract...")
            except FileNotFoundError:
                raise "No contract found in save directory"

            self.load_from_save = True

        # create new game save directory
        else:
            if save_dir is None:
                self.save_dir = f"saves/game_save_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            else:
                self.save_dir = save_dir
            U.f_mkdir(self.save_dir)
            U.f_mkdir(f"{self.save_dir}/episodes")

        # create judge
        self.judge = Voyager(
            server_port=server_port,
            username=self.judge_username,
            ckpt_dir=f"{self.save_dir}/{self.judge_username}_ckpt",
            episode_timeout=episode_timeout,
            **options
        )
        self.judge.env.reset()

        # create agents
        for i in range(num_agents):
            username = self.usernames[i]
            ckpt_dir = f"{self.save_dir}/{username}_ckpt"

            agent = Voyager(
                username=username,
                server_port=server_port + 1 + i,
                ckpt_dir=ckpt_dir,
                episode_timeout=episode_timeout,
                **options
            )
            self.agents.append(agent)

    def run_threads(self, target, args=None, include_judge=False, shared_args=False):
        """
        Runs target function in parallel for each agent. args is a dictionary of arguments to pass to each thread,
        where the key is the agent's username.

        For example,
        args = {'Voyager3000': {'arg1': 1, 'arg2': 2}, 'Voyager3001': {'arg1': 3, 'arg2': 4}}
        """
        logging.info(f"Starting threads for agents with target function: {target}")
        agents = self.agents + [self.judge] if include_judge else self.agents
        logging.debug(f"Agents included for threading: {[agent.username for agent in agents]}")

        if args is None:
            args = {agent.username: {} for agent in agents}
            logging.debug("No arguments provided; using empty dictionaries for all agents.")
        if shared_args:
            args = {agent.username: args for agent in agents}
            logging.debug("Using shared arguments for all agents.")

        results = {}
        threads = []
        for agent in agents:
            result = {}
            logging.debug(f"Arguments for agent {agent.username}:\n{args[agent.username]}")
            thread = threading.Thread(target=target, args=(agent, result), kwargs=args[agent.username], daemon=True)
            results[agent.username] = result
            threads.append(thread)
            thread.start()
            logging.info(f"Thread started for agent {agent.username}")
        for thread in threads:
            thread.join()
            logging.info(f"Thread for agent {thread.name} has completed")
        logging.info(f"All threads have completed. Results:\n{results}")
        return results

    def reset_agents(self, mode='soft', timeout=10):
        args = {agent.username: {'options': {'mode': mode, 'wait_ticks': agent.env_wait_ticks}} for agent in
                self.agents}
        self.run_threads(lambda agent, _, options: agent.env.reset(options=options), args=args)
        time.sleep(2)

    def save_scenario(self, save_options):
        """
        Saves the current scenario to a json file. The scenario is saved as a dictionary with the following keys:
        - block_positions: a dictionary of block types and their positions
        - spawn_locations: a dictionary of agent usernames and their spawn locations
        - chest_contents: a string of the chest contents in minecraft format
        """
        print('Saving scenario...')

        if len(self.agents) == 0:
            raise 'At least one agent must be initialized to save scenario'

        scenario_block_types = save_options['scenario_block_types']
        file_name = save_options['file_name']
        center_position = save_options['center_position']
        remove_blocks = save_options['remove_blocks']

        # set file_name
        if self.scenario_file != file_name:
            print(f'Warning: scenario_file does not match file_name, using {file_name}')
        file_name = "scenarios/" + file_name

        def extract_block_positions(events_ar):
            block_types = scenario_block_types
            block_positions_ar = {block: [] for block in block_types}

            for event in events_ar:
                if event[0] == 'onChat':
                    message = event[1]['onChat']
                    # Checking each block type
                    for block in block_types:
                        if block in message:
                            # Extracting positions
                            positions = message.split(f'{block}: ')[-1].replace('),(', ');(').split(';')
                            for pos in positions:
                                if not pos.strip():  # Check if the position is an empty string
                                    continue
                                x_el, y_el, z_el = map(int, pos.strip('()').split(', '))
                                coord_dict = {'x': x_el, 'y': y_el, 'z': z_el}  # Convert coords to dictionary format
                                block_positions_ar[block].append(coord_dict)

            # Removing block types with no positions found
            block_positions_ar = {k: v for k, v in block_positions_ar.items() if v}
            return block_positions_ar

        # self.judge.env.reset(
        #         options={
        #             "mode": "hard",
        #             "wait_ticks": self.judge.env_wait_ticks,
        #         }
        #     )

        x, y, z = center_position['x'], center_position['y'], center_position['z']

        # Remove blocks of type scenario_block_types, so they don't interfere with the scenario
        if remove_blocks:
            input(f"Center position is set to {center_position}. Blocks of type {scenario_block_types} will be "
                  f"deleted nearby. Press enter to continue...")
            print("Removing blocks...\n")
            self.judge.env.step(
                f"await bot.chat('/tp {x} {y} {z}');"
                + U.remove_blocks_commands(scenario_block_types, center_position),
                programs=self.judge.skill_manager.programs,
            )

        # Save blocks of type scenario_block_types
        input(f"Construct the scenario. Blocks of type {scenario_block_types} will be saved. Press enter when done...")
        print("Saving blocks...\n")
        events = self.judge.env.step(
            f"bot.chat('/tp {x} {y} {z}');"
            + f"await getBlockPositions(bot, {U.json_dumps(scenario_block_types)}, {U.json_dumps(center_position)})",
            # should be able to specify center square of save area
            programs=self.judge.skill_manager.programs,
        )

        block_positions = extract_block_positions(events)

        # save block_positions as well as default spawn locations and chest contents
        json_contents = {
            'description': 'There is a chest with a diamond pickaxe.',
            'secret_description': 'Agents do not see this description, just for information',
            'tasks': {self.usernames[0]: 'mine diamond', self.usernames[1]: 'mine iron'},
            'center_position': center_position,
            'block_positions': {'facing': 'north', **block_positions},
            'spawn_locations': {self.usernames[0]: {'x': x + 1, 'y': y, 'z': z + 1},
                                self.usernames[1]: {'x': x - 1, 'y': y, 'z': z - 1}},
            'reward_item_name': ['diamond'],
            'chest_contents': {'diamond_pickaxe': 1},
        }
        U.custom_dump(json_contents, file_name)
        print('Scenario saved in ', file_name)
        self.judge.close()

    def load_scenario(self, reset='soft'):
        try:
            file_name = "scenarios/" + self.scenario_file
            json_contents = U.json_load(file_name)
            logging.info(f'Loading scenario file: {self.scenario_file}...')
        except FileNotFoundError:
            logging.error(f'No scenario file found: {self.scenario_file}')
            raise FileNotFoundError('No scenario file found')

        self.scenario_description = json_contents['description']
        tasks = json_contents['tasks']
        center_position = json_contents['center_position']
        block_positions = json_contents['block_positions']
        spawn_locations = json_contents['spawn_locations']
        chest_contents = U.parse_chest_contents(json_contents['chest_contents'])
        self.reward_item_names = json_contents['reward_item_names']
        scenario_block_types = list(block_positions.keys())
        scenario_block_types.remove('facing')
        self.chest_memory = {}

        logging.info('Scenario description and details loaded.')

        for i, agent in enumerate(self.agents):
            agent.task = tasks[agent.username]
            logging.info(f'Set task for agent {agent.username}: {agent.task}')

        self.judge.task = tasks
        logging.info('Set judge tasks.')

        js_file = file_name.replace('.json', '.js')
        if U.f_exists(js_file):
            self.scenario_code = U.load_text(js_file)
            logging.info(f'Loaded scenario code from: {js_file}')
        else:
            logging.warning(f'No scenario code file found for: {js_file}')

        if len(self.agents) == 0:
            logging.error('At least one agent must be initialized to load scenario')
            raise ValueError('At least one agent must be initialized to load scenario')

        self.reset_agents(mode='hard')
        logging.info('Agents reset.')

        x, y, z = center_position['x'], center_position['y'], center_position['z']

        logging.info('Setting up environment...')
        self.judge.env.step(
            f"bot.chat('/gamemode spectator {self.judge_username}');"
            + f"bot.chat('/tp {self.judge_username} {x} {y + 20} {z}');"
            + f"bot.chat('/gamerule randomTickSpeed 3');"
            + f"bot.chat('/gamerule spawnRadius 0');"
            + U.remove_drops_commands()
            + (U.remove_blocks_commands(scenario_block_types, center_position) if reset == 'hard' else '')
            + U.spawn_commands(self.usernames, spawn_locations)
            + U.add_block_commands(block_positions)
            + U.chest_commands(block_positions, chest_contents),
            programs=self.judge.skill_manager.programs,
        )
        logging.info('Environment setup complete.')

        if self.scenario_code is not None:
            self.judge.env.step(self.scenario_code)
            logging.info('Executed scenario code.')

    # update a global chest memory to keep consistent across agents
    def update_chest_memory(self, chests):
        for position, chest in chests.items():
            if position in self.chest_memory:
                if isinstance(chest, dict):
                    self.chest_memory[position] = chest
                if chest == "Invalid":
                    print(f"Removing chest {position}: {chest}")
                    self.chest_memory.pop(position)
            else:
                if chest != "Invalid":
                    print(f"Saving chest {position}: {chest}")
                    self.chest_memory[position] = chest

        # update agent chest memories
        for agent in self.agents + [self.judge]:
            agent.action_agent.chest_memory = self.chest_memory

    def check_task_success(self, events, max_retries=5):
        def ai_check_task_success(agent, result, events):
            logging.info(f"Starting task success check for agent: {agent.username}")
            if agent.username == self.judge_username:
                critic_agent = agent.judge_agent
            else:
                critic_agent = agent.critic_agent

            logging.debug(f"Using critic agent:\n{critic_agent}")

            human_message = critic_agent.render_human_message(
                events=events,
                task=agent.task,
                scenario=self.scenario_description,
                contract=self.contract,
                context=agent.context,
                chest_observation=agent.action_agent.render_chest_observation())

            logging.debug(f"Human message rendered:\n{human_message}")

            messages = [critic_agent.render_system_message(),
                        human_message]

            logging.debug(f"Messages prepared for AI check:\n{messages}")

            critic_response = critic_agent.ai_check_task_success(messages=messages, max_retries=max_retries)

            logging.debug(f"Critic response received:\n{critic_response}")

            if agent.username == self.judge_username:
                emeralds, critique = critic_response
                success = None
                logging.debug(f"Judge response:\n{critique},\nEmeralds:\n{emeralds}")
            else:
                success, critique = critic_response
                emeralds = None
                logging.debug(f"Agent success status:\n{success},\nCritique:\n{critique}")

            result.update({'success': success, 'critique': critique, 'emeralds': emeralds})

        # TODO: include judge human feedback
        def human_check_task_success():
            results = {agent.username: {} for agent in self.agents}
            # log critic human critic messages
            for agent in self.agents:
                agent.critic_agent.render_human_message(
                    events=events[agent.username]['events'],
                    task=agent.task,
                    scenario=self.scenario_description,
                    contract=self.contract,
                    context=agent.context,
                    chest_observation=agent.action_agent.render_chest_observation(),
                )
            # collect critiques about agents
            for agent in self.agents:
                confirmed = False
                success = False
                critique = ""
                while not confirmed:
                    success = input(f"{agent.username} Success? (y/n)")
                    success = success.lower() == "y"
                    critique = input("Enter your critique:")
                    print(f"Success: {success}\nCritique: {critique}")
                    confirmed = input("Confirm? (y/n)") in ["y", ""]
                results[agent.username].update({'success': success, 'critique': critique})
            return results

        if self.critic_mode == "manual":
            return human_check_task_success()

        return self.run_threads(ai_check_task_success, args=events, include_judge=True)

    def save_episode(self, results):
        U.dump_json(results, f"{self.save_dir}/episodes/episode{self.episode}/code.json")

    def load_episode(self, episode):
        if not isinstance(episode, int):
            raise ValueError("episode must be an integer")

        file_name = f"{self.save_dir}/episodes/episode{episode}/code.json"
        json_contents = U.json_load(file_name)
        return json_contents

    # def replay_episode(self, episode):
    #     if not isinstance(episode, int):
    #         raise ValueError("episode must be an integer")

    #     episode_results = self.load_episode(episode)
    #     self.run_threads(env_step, args=episode_results)

    def run_episode(self, episode=None, reload=True, reset='soft'):
        # get ai_message and parse
        def get_ai_message_parse(agent, result):
            logging.debug(f"Getting AI message for agent {agent.username}")
            if agent.action_agent_rollout_num_iter < 0:
                raise ValueError("Agent must be reset before stepping")

            ai_message = agent.action_agent.llm(agent.messages)
            logging.info(f"AI message for {agent.username}:\n{ai_message.content}")
            agent.conversations.append(
                (agent.messages[0].content, agent.messages[1].content, ai_message.content)
            )
            parsed_result = agent.action_agent.process_ai_message(message=ai_message)
            result.update({'parsed_result': parsed_result})

        # do env.step
        def env_step(agent, result, parsed_result):
            logging.debug(f"Executing environment step for {agent.username}")
            if not isinstance(parsed_result, dict):
                assert isinstance(parsed_result, str)
                logging.error(f"Invalid parsed result type for {agent.username}:\n{parsed_result}")
                agent.recorder.record([], agent.task)

            code = parsed_result["program_code"] + "\n" + parsed_result["exec_code"]
            events_ar = agent.env.step(
                f"await saveRewards(bot, {U.json_dumps(self.reward_item_names)}, "
                + f"'{self.save_dir}/episodes/episode{self.episode}');"
                + code,
                programs=agent.skill_manager.programs,
            )
            agent.recorder.record(events_ar, agent.task)  # what is this for??
            self.update_chest_memory(events_ar[-1][1]["nearbyChests"])
            result.update({'events': events_ar})

        # update messages for next round
        def update_agent(agent, result, parsed_result, events, success, critique, contract_critique, emeralds):
            logging.debug(f"Updating agent {agent.username}")
            new_skills = agent.skill_manager.retrieve_skills(
                query=agent.context + "\n\n" + agent.action_agent.summarize_chatlog(events)
            )
            system_message = agent.action_agent.render_system_message(skills=new_skills)
            human_message = agent.action_agent.render_human_message(
                events=events,
                code=parsed_result["program_code"],
                task=agent.task,
                contract=agent.contract,
                scenario=agent.scenario,
                context=agent.context,
                critique=critique,
                contract_critique=contract_critique,
            )
            agent.last_events = copy.deepcopy(events)
            agent.messages = [system_message, human_message]
            assert len(agent.messages) == 2
            agent.action_agent_rollout_num_iter += 1

            done = (agent.action_agent_rollout_num_iter >= agent.action_agent_task_max_retries or success)
            info = {
                "task": agent.task,
                "success": success,
                "conversations": agent.conversations,
                "emeralds": emeralds
            }
            if success:
                assert (
                        "program_code" in parsed_result and "program_name" in parsed_result
                ), "program and program_name must be returned when success"
                info["program_code"] = parsed_result["program_code"]
                info["program_name"] = parsed_result["program_name"]

            agent.logger(
                f"****Action Agent human message****\n{agent.messages[-1].content}"
            )
            result.update({'messages': agent.messages, 'done': done, 'info': info})

        # replace chat events with those from the agent who lived longest and save both players observations
        # note: this is a hacky solution to a problem that should be fixed in the future
        def fix_chat_events(events_ar):
            logging.info(f"Entering fix_chat_events")

            # collect all chat events for each agent
            chat_events = {agent.username: [] for agent in self.agents}
            other_events = {agent.username: [] for agent in self.agents}
            for agent, other_agent in [self.agents, self.agents[::-1]]:  # wont work if num_agents != 2
                agent_events = events_ar.get(agent.username, {}).get('events', [])
                if not agent_events:
                    logging.warning(f"fix_chat_events function: Agent events is empty for {agent.username}")

                for (event_type, event) in agent_events:
                    if event_type == 'onChat':
                        chat_events[agent.username].append((event_type, event))
                    # record both agents observations for reading inventory etc
                    elif event_type == 'observe':
                        other_events[other_agent.username].insert(0, ('otherObserve', event))
                        other_events[agent.username].append((event_type, event))
                    else:
                        other_events[agent.username].append((event_type, event))

            logging.debug(f"Chat events collected:\n{chat_events}")
            logging.debug(f"Other events collected:\n{other_events}")

            # copy in the longest thread of chats
            longest_thread = max(chat_events.values(), key=len)
            new_events = {agent.username: {'events': longest_thread + other_events[agent.username]} for agent in
                          self.agents}

            # copy one of the agents events for the judge
            new_events[self.judge_username] = new_events[self.agents[0].username]
            logging.debug(f"Final reorganized events:\n{new_events}")
            return new_events

        logging.info(f"Starting run_episode with reload={reload}, reset={reset}, episode={episode}")

        # reset for both agents and load scenario
        if reload:
            logging.info("Reloading scenario...")
            self.load_scenario(reset=reset)
            # time.sleep(3) # wait for voyagers and scenario to load

        # if a specific episode is provided, look for contract and play it
        # ideally this should be moved to a different function (except env_step should be moved too)
        if episode is not None:
            if not isinstance(episode, int):
                raise ValueError("episode must be an integer")
            logging.info(f"Loading specific episode: {episode}")
            episode_results = self.load_episode(episode)
            self.run_threads(env_step, args=episode_results)
            self.reset_agents()
            return

        # get ai_message and parse in parallel
        logging.info('Processing AI messages and parsing results...')
        parsed_results = self.run_threads(get_ai_message_parse)

        # save episode
        self.save_episode(parsed_results)

        # do env.step in parallel`
        logging.info('Executing environment steps based on parsed AI messages...')
        events = self.run_threads(env_step, args=parsed_results)

        self.reset_agents()

        logging.info('Fixing chat events...')
        events = fix_chat_events(events)
        # check for task success
        logging.info('Checking task success for the episode...')

        critic_response = self.check_task_success(events)

        logging.info(f"Task success response:\n{critic_response}")

        # update agents (note this function does not need to be run with threads; could add a flag to just iterate)
        results = self.run_threads(update_agent, args={
            agent.username: {
                **parsed_results[agent.username],
                **events[agent.username],
                **critic_response[agent.username],
                'contract_critique': critic_response[self.judge.username]['critique'][agent.username],
                'emeralds': critic_response[self.judge.username]['emeralds'][agent.username],
            } for agent in self.agents})

        return results

    def negotiate_contract(self, max_turns=8):
        """
        Generates a contract for the agents to follow and sets self.contract to the contract.
        """
        logging.info('Negotiating contract...')

        if self.scenario_description is None:
            raise ValueError("Scenario must be loaded before negotiating contract")

        agent1 = self.agents[0]
        agent2 = self.agents[1]

        negotiator1 = Negotiator(
            name=agent1.username,
            task=agent1.task,
            other_name=agent2.username,
            other_task=agent2.task,
            scenario=self.scenario_description,
            model=self.negotiator_model_name,
            temperature=self.negotiator_temperature,
        )

        negotiator2 = Negotiator(
            name=agent2.username,
            task=agent2.task,
            other_name=agent1.username,
            other_task=agent1.task,
            scenario=self.scenario_description,
            model=self.negotiator_model_name,
            temperature=self.negotiator_temperature,
        )

        # hold a negotiation between players, where negotiator1 starts first
        negotiation = Negotiation(negotiator1, negotiator2, max_turns=max_turns, save_dir=self.save_dir)
        negotiation.simulate()
        self.contract = negotiation.get_contract()

    def run(self):
        if self.load_from_save:
            input("Warning: loaded from saved directory. Continuing may overwrite saved files. "
                  "Press enter to continue...")

        logging.info('Loading scenario...')
        self.load_scenario(reset='hard')

        if self.contract_mode == "auto":
            if self.contract is not None:
                logging.warning("Contract provided but contract_mode is 'auto'. Contract will be ignored.")
            logging.info('Negotiating contract...')
            self.negotiate_contract()
        elif self.contract_mode == "manual":
            logging.info('Contract is provided manually')

        contract_path = f"{self.save_dir}/contract.txt"
        with open(contract_path, 'w') as contract_file:
            logging.info(f'Saving the contract to {contract_path}...')
            contract_file.write(self.contract)

        logging.info('Initializing threads for agents...')
        self.run_threads(
            target=lambda agent_t, _, args: agent_t.reset(task=agent_t.task, **args),
            args={
                'args': {
                    'contract': self.contract,
                    'scenario': self.scenario_description,
                    'context': "",
                    'reset_env': False,
                }
            },
            shared_args=True
        )
        logging.info('Threads initialized.')

        replay = False
        done = False

        while not done:
            if replay:
                logging.info('Repeating episode...')
                self.run_episode(episode=self.episode, reload=True, reset='soft')
            else:
                episode_dir = f"{self.save_dir}/episodes/episode{self.episode}"
                U.f_mkdir(episode_dir)

                reload = self.episode != 0
                logging.info(f'Starting episode {self.episode}...')
                results = self.run_episode(reload=reload, reset='soft')
                logging.info(f'Episode {self.episode} completed.')

                for agent in self.agents:
                    emeralds = results[agent.username]['info']['emeralds']
                    logging.info(f"{agent.username} has {emeralds} emeralds.")

                emeralds_data = {agent.username: results[agent.username]['info']['emeralds'] for agent in self.agents}
                U.json_dump(emeralds_data, f"{episode_dir}/emeralds.json")

            if self.continuous:
                self.episode += 1
                logging.info(f'Episode {self.episode} completed, moving to next episode.')
                if self.episode == self.num_episodes:
                    done = True
                    logging.info('Reached the maximum number of episodes. Ending simulation.')
            else:
                user_response = input("Press enter to continue or 'r' to repeat...")
                if user_response == 'r':
                    replay = True
                    logging.info('User chose to repeat the episode.')
                else:
                    replay = False
                    self.episode += 1
                    logging.info(f'User chose to continue. Preparing episode {self.episode}.')

        logging.info('Simulation completed. Exiting...')

    def close(self):
        server = self.judge.env.server
        _ = requests.post(f"{server}/stop")
        for agent in self.agents + [self.judge]:
            agent.env.mineflayer.stop()
