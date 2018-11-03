""" Module with configuration for Walker environment """

import os
import inspect

CURRENT_DIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
PARENT_DIR = os.path.dirname(os.path.dirname(CURRENT_DIR))
os.sys.path.insert(0, PARENT_DIR)

WALKER = "Walker2DBulletEnv-v0"
MODEL = "human"
RENDER_ENV = False

MONITOR_PATH = 'C:\\Users\\Marlena\\Desktop\\inzynierka\\rainforcement learning\\Walker\\motor'
