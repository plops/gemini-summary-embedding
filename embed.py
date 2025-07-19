#!/usr/bin/env python3
#
# /// script
# dependencies = [
#   "sqlite-minutils",
#   "google-genai",
# ]
# ///

from sqlite_minutils.db import *
from google import genai


# %% Load the database with video summaries
db = Database("out.db")
