import os.path
import time
import warnings
from typing import Any, Tuple, Dict
import random

import requests
import json

import gymnasium as gym
from gymnasium.core import ObsType

import voyager.utils as U

from .minecraft_launcher import MinecraftInstance
from .process_monitor import SubprocessMonitor

import logging


class VoyagerEnv(gym.Env):
    def __init__(
            self,
            mc_port=None,
            username="bot",
            azure_login=None,
            server_host="http://127.0.0.1",
            server_port=3000,
            request_timeout=5,
            log_path="./logs",
    ):
        if not mc_port and not azure_login:
            raise ValueError("Either mc_port or azure_login must be specified")
        if mc_port and azure_login:
            warnings.warn(
                "Both mc_port and mc_login are specified, mc_port will be ignored"
            )
        self.mc_port = mc_port
        self.username = username
        self.azure_login = azure_login
        self.server = f"{server_host}:{server_port}"
        self.server_port = server_port
        self.request_timeout = request_timeout
        self.log_path = log_path
        self.mineflayer = self.get_mineflayer_process(server_port)
        if azure_login:
            self.mc_instance = self.get_mc_instance()
        else:
            self.mc_instance = None
        self.has_reset = False
        self.reset_options = None
        self.connected = False
        self.server_paused = False

    def get_mineflayer_process(self, server_port):
        U.f_mkdir(self.log_path, f"mineflayer/port_{str(server_port)}")
        file_path = os.path.abspath(os.path.dirname(__file__))
        return SubprocessMonitor(
            commands=[
                "node",
                U.f_join(file_path, "mineflayer/index.js"),
                str(server_port),
            ],
            name="mineflayer",
            ready_match=r"Server started on port (\d+)",
            log_path=U.f_join(self.log_path, f"mineflayer/port_{str(server_port)}"),
            username=self.username
        )

    def get_mc_instance(self):
        print("Creating Minecraft server")
        U.f_mkdir(self.log_path, "minecraft")
        return MinecraftInstance(
            **self.azure_login,
            mineflayer=self.mineflayer,
            log_path=U.f_join(self.log_path, "minecraft"),
        )

    def send_request(self, url, json_data=None, timeout=None, max_retries=3, backoff_factor=1):
        """
        Send a request to the specified URL, retrying up to max_retries times with exponential backoff.
        Args:
            url (str): The URL to which the request is sent.
            json_data (dict, optional): The JSON data to send in the request. Defaults to None.
            timeout (int): Timeout for the request.
            max_retries (int): Maximum number of retries if the request fails.
            backoff_factor (int): Factor by which to multiply the wait time for each retry.
        Returns:
            requests.Response: The response object from the server.
        """
        effective_timeout = timeout or self.request_timeout
        attempts = 0

        while attempts < max_retries:
            try:
                logging.info(f"Attempt {attempts + 1}: Sending request to {url}")
                response = requests.post(url, json=json_data, timeout=effective_timeout)
                response.raise_for_status()  # Raises an HTTPError for bad responses
                logging.info("Received response successfully.")
                return response
            except requests.exceptions.HTTPError as errh:
                logging.error("HTTP Error: %s", errh)
            except requests.exceptions.ConnectionError as errc:
                logging.error("Error Connecting: %s", errc)
            except requests.exceptions.Timeout as errt:
                logging.error("Timeout Error: %s", errt)
            except requests.exceptions.RequestException as err:
                logging.error("Request Error: %s", err)

            attempts += 1
            sleep_time = backoff_factor * (2 ** attempts) + random.uniform(0, 1)
            logging.info(f"Retrying in {sleep_time:.2f} seconds...")
            time.sleep(sleep_time)

        logging.error("Failed to receive a valid response after several attempts.")
        return None

    def start_mc_instance(self):
        logging.debug("Starting Minecraft server")
        self.mc_instance.run()
        self.mc_port = self.mc_instance.port
        self.reset_options["port"] = self.mc_instance.port
        logging.debug(f"Server started on port {self.reset_options['port']}")

    def restart_mineflayer_with_backoff(self):
        retry, max_retries, backoff_factor = 0, 3, 2
        while retry <= max_retries:
            logging.debug(f"Attempt {retry + 1}: Mineflayer process has exited, attempting to restart")
            self.mineflayer.run()
            sleep_time = backoff_factor ** retry + random.uniform(0, 1)  # Exponential backoff with jitter
            logging.info(f"Retrying in {sleep_time:.2f} seconds...")
            time.sleep(sleep_time)
            if self.mineflayer.is_running:
                logging.info("Mineflayer restarted successfully.")
                return
            else:
                logging.warning(f"Mineflayer failed to start on attempt {retry + 1}")
            retry += 1
        logging.error("Failed to restart Mineflayer after several attempts")
        raise RuntimeError("Failed to restart Mineflayer after several attempts")

    def try_server_start_endpoint(self):
        try:
            result = self.send_request(f"{self.server}/start", json_data=self.reset_options)
            if result is not None:
                if result.status_code == 200:
                    logging.debug("Server started successfully")
                    return result.json()
            else:
                logging.warning("Received non-200 status code")
        except Exception as e:
            logging.error(f"Server start failed. Error: {str(e)}")
        raise RuntimeError("Failed to start server via /start endpoint")

    def check_process(self):
        if self.mc_instance and not self.mc_instance.is_running:
            self.start_mc_instance()

        if not self.mineflayer.is_running:
            self.restart_mineflayer_with_backoff()

        retry, max_retries = 0, 1
        while retry <= max_retries:
            try:
                results = self.try_server_start_endpoint()
                return results
            except Exception as e:
                logging.error(f"check_process: Server failed to /start: {str(e)}. Attempts: {max_retries - retry}")

                if retry < max_retries:
                    logging.debug(f"check_process: Restarting Mineflayer due to an error")
                    self.restart_mineflayer_with_backoff()
                retry += 1

    def step(self, code: str, programs: str = ""):
        if not self.has_reset:
            raise RuntimeError("Environment has not been reset yet")
        self.check_process()
        data = {"code": code, "programs": programs}
        result = self.send_request(f"{self.server}/step", data)
        if result is None:
            raise RuntimeError(f"Failed to get a valid response from the Minecraft server, port: {self.server_port}. "
                               f"Server is running: {self.mineflayer.is_running}")
        returned_data = result.json()
        return json.loads(returned_data)

    def render(self):
        raise NotImplementedError("render is not implemented")

    def reset(self, *, seed=None, options=None) -> Tuple[ObsType, Dict[str, Any]]:
        if options is None:
            options = {}

        # Validate the options dictionary to ensure correct settings are applied
        if options.get("inventory", {}) and options.get("mode", "hard") != "hard":
            raise RuntimeError("Inventory can only be set when the reset mode is 'hard'.")

        # Setting up reset options based on the provided arguments and defaults
        self.reset_options = {
            "port": self.mc_port,
            "username": self.username,
            "reset": options.get("mode", "hard"),
            "inventory": options.get("inventory", {}),
            "equipment": options.get("equipment", []),
            "spread": options.get("spread", False),
            "waitTicks": options.get("wait_ticks", 5),
            "position": options.get("position", None),
        }

        # Ensure mineflayer is properly shutdown before attempting a reset
        self.mineflayer.stop()
        time.sleep(1)  # Ensures the process has time to terminate properly

        # Re-initialize or check server/mineflayer process status
        returned_data = self.check_process()
        if returned_data is None:
            raise RuntimeError("Failed to start or reset the server process properly.")

        self.has_reset = True
        self.connected = True

        # All future resets after the initial setup will be of type 'soft'
        self.reset_options["reset"] = "soft"

        return json.loads(returned_data)

    def close(self):
        if self.connected:
            result = self.send_request(f"{self.server}/stop")
            if result.status_code == 200:
                self.connected = False
        if self.mc_instance:
            self.mc_instance.stop()

    def pause(self):
        if self.mineflayer.is_running and not self.server_paused:
            result = self.send_request(f"{self.server}/pause")
            if result.status_code == 200:
                self.server_paused = True
        return self.server_paused

    def unpause(self):
        if self.mineflayer.is_running and self.server_paused:
            result = self.send_request(f"{self.server}/pause")
            if result is not None:
                if result.status_code == 200:
                    self.server_paused = False
                else:
                    print(result.json())
        return self.server_paused

    def get_inventory(self):
        result = self.send_request(f"{self.server}/inventory")
        if result is None:
            raise RuntimeError("Failed to get a valid response from the Minecraft server")
        return result.json()['inventory']

    def get_observations(self):
        result = self.send_request(f"{self.server}/observe")
        if result is None:
            raise RuntimeError("Failed to get a valid response from the Minecraft server")
        returned_data = result.json()
        return json.loads(returned_data)

    def give_item(self, item, target, count=1):
        data = {"item": item, "target": target, "count": count}
        result = self.send_request(f"{self.server}/give-item", json_data=data)
        if result is None:
            raise RuntimeError("Failed to give item to the bot")
        return result.json()
