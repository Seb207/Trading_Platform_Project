import sys, os
import json

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import datetime as dt

import refinitiv.data as rd
import pandas_datareader as pdr

# ==============================================================================
# 1. API Session Management
# ==============================================================================

class Factor_Management():
    def __init__(self, filepath):
        self.filepath = filepath

    def key_loader(self):
        """
        loads app key from the config file
        :return: app_key
        """
        config_file_path = self.filepath
        try:
            # Open JSON file
            with open(config_file_path, 'r') as f:
                config_data = json.load(f)

            app_key = config_data['sessions']['platform']['rdp']['app-key']

            print(f"Key is successfully loaded")

        except FileNotFoundError:
            print(f"Error: '{config_file_path}' File not found.")

        except KeyError as e:
            print(f"Error: No key {e} in file.")

        return app_key

    def check_state(self, state, message, session):
        print(f"State: {state}")
        print(f"Message: {message}")
        print("\n")

    def refinitiv_open_session(self):
        """
        function to open Refinitiv API session

        return
        ------
        Refinitiv API session state
        """
        session = rd.session.desktop.Definition(
            app_key=self.key_loader()).get_session()
        rd.session.set_default(session)

        session.on_state(self.check_state)

        session.open()

        return session.open_state, session

    def refinitiv_close_session(self):
        """
        function to close Refinitv API session

        return
        ------
        state: closed session
        """
        session = rd.session.get_default()

        session.on_state(self.check_state)

        session.close()

        return session.open_state

    def price_data_loader(self, ric, start, end, interval):
        """
        Loads price data according to the arguments

        :param ric: RIC code for the target instrument
        :param start: start date of the data
        :param end: end date of the data
        :param interval: interval of the data
        :return: data_loaded = pd.DataFrame()
        """
        if interval == 'minute' or interval == '1min' or interval == '1h' or interval == '1s':
            data_loaded = pd.DataFrame(
                rd.get_history(ric, start=start, end=end, interval=interval, fields=['TRDPRC_1']))
        else:
            data_loaded = pd.DataFrame(
                rd.get_history(ric, start=start, end=end, interval=interval, fields=['TR.CLOSE']))

        return data_loaded

    def volume_data_loader(self, ric, start, end, interval):
        """
        Loads volume data according to the arguments

        :param ric: RIC code for the target instrument
        :param start: start date of the data
        :param end: end date of the data
        :param interval: interval of the data
        :return: data_loaded = pd.DataFrame()
        """
        if interval == 'minute' or interval == '1min' or interval == '1h' or interval == '1s':
            data_loaded = pd.DataFrame(
                rd.get_history(ric, start=start, end=end, interval=interval, fields=['ACVOL_UNS']))
        else:
            data_loaded = pd.DataFrame(
                rd.get_history(ric, start=start, end=end, interval=interval, fields=['TR.Volume']))

        return data_loaded

    def fi_data_loader(self, ric, start, end, interval, if_yield):
        """
        Loads fixed income data according to the arguments
        :param ric: RIC code for the target instrument
        :param start: start date of the data
        :param end: end date of the data
        :param interval: interval of the data
        :param if_yield: boolean whether it is for price or yield
        :return: pd.DataFrame()
        """
        if if_yield:
            field = 'MID_YLD_1'
        else:
            field = 'MID_PRICE'

        data_loaded = pd.DataFrame(rd.get_history(ric, start=start, end=end, interval=interval, fields=[field]))

        return data_loaded

    def get_fred_data(self, series, start, end):
        """
        Loads FRED data according to the arguments

        :param series: FRED series code for the target instrument
        :param start: start date of the data
        :param end: end date of the data
        :return: data_loaded = pd.DataFrame()
        """
        data_loaded = pdr.get_data_fred(series, start=start, end=end)

        return data_loaded