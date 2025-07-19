#!/usr/bin/env python3
#
# /// script
# dependencies = [
#   "sqlite-minutils",
#   "google-genai",
# ]
# ///

# open the sqlite database out.db
from sqlite_minutils.db import *
from google import genai

db = Database("out.db")
