import torch
import transformers
import json
import os, platform

import irequest as ireq
import irequest.composition as comp

workdir = os.path.dirname(__file__)
ireq.Dev()