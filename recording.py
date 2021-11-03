import os
import sys
from pathlib import Path
import subprocess
import time


def start(cmd):
    subprocess.Popen([sys.executable, cmd], start_new_session=True)

os.chdir(str(Path(__file__).parent / 'deformationcytometer' / 'recording'))
start('LiveDisplay.py')
start('store_images.py')
start('streamAcquisition.py')
while True:
    time.sleep(10)
