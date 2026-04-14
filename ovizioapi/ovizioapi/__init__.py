import os
import sys
import clr
from pathlib import Path
from copy import deepcopy
from deepdiff import DeepDiff
from typing import Tuple

from ovizioapi.config import settings, read_xml_file, default_reconstruction_config

executable_path = settings.paths.osone + "\\bin"
#config_path = settings.paths.osone + "\\config"

print("\n[ INIT ] Initialize Ovizio API...")
print("[ INFO ] Using OsOne Path: %s" % executable_path)

#############################
###       DLL Setup       ###
#############################
# Add Ovizio bin folder to path
sys.path.append(executable_path)
# Test Version (Windows only)
if os.name == "nt":
    from .base import get_osone_version

    version = get_osone_version()
    print(f"[ INFO ] OsOne Version: {version}")
else:
    print("[ ERRO ] Could not determine OsOne Version")
# Register DLLs
clr.AddReference("OvizioApiNet")
clr.AddReference("OvizioCoreWrapper")
# Perform Imports
import OvizioApiNet.Computations
import OvizioApiNet.Image
import OvizioCoreWrapper
import OvizioApiNet

#############################
### Convenience functions ###
#############################
def get_application_config_path() -> Path:
    return Path(OvizioApiNet.OvizioApiNet.get_ApplicationConfigurationPath())


def get_experiments_path() -> Path:
    return Path(OvizioApiNet.OvizioApiNet.get_ExperimentsPath())


def get_user_config_path() -> Path:
    return Path(OvizioApiNet.OvizioApiNet.get_UserConfigurationPath()) / "Config"


def get_user_reconstruction_config_path() -> Path:
    return Path(OvizioApiNet.OvizioApiNet.get_UserReconstructionParametersFile())


def get_use_gpu() -> bool:
    return OvizioApiNet.OvizioApiNet.get_UseGPU()


def set_use_gpu(value: bool):
    OvizioApiNet.OvizioApiNet.set_UseGPU(value)


def get_active_config():
    user_config_path = get_user_config_path() / "DefaultReconstructionParameters.xml"
    application_config_path = get_application_config_path() / "DefaultReconstructionParameters.xml"

    if not application_config_path.is_file():
        raise FileNotFoundError(f"Application config '{application_config_path}' not found!")

    # Get Application Default
    application_config = read_xml_file(application_config_path, "ovizio")
    config = deepcopy(application_config)
    diff = {}

    # Update defaults if user specific config exists
    if user_config_path.is_file():
        user_config = read_xml_file(user_config_path, "ovizio")
        config.update(user_config)
        diff = DeepDiff(application_config, config)

    return config, diff


def validate_config() -> Tuple[bool, dict]:
    config, _ = get_active_config()
    diff = DeepDiff(default_reconstruction_config, config)
    return not diff, diff

#############################
###     Initialzation     ###
#############################
# OvizioApiNet.OvizioApiNet.Initialize()
# Params
# applicationConfigurationPath: str
# userConfigurationPath: str
# experimentsPath: str
# redirectLogToCout: bool
#
# Empty Initialize() will use the default locations
# Application folder: C:\Program Files\Ovizio\OsOne
# User configuration folder: C:\users\<currentuser>\appdata\Roaming\OsOne
OvizioApiNet.OvizioApiNet.Initialize(redirectLogToCout=settings.redirectLog)

#############################
###          GPU          ###
#############################
# Enable / Disable GPU Usage
print(f"[ INIT ] Setting GPU usage to '{settings.useGPU}'")
set_use_gpu(settings.useGPU)

#############################
###      Check Config     ###
#############################
print("[ INIT ] Checking Config Files")
valid, diff = validate_config()
if valid:
    print("[  OK  ] Config Files Are Valid")
else:
    print("[ WARN ] Config Files Are INVALID!")
    raise Warning(f"Config Devivation Detected: '{diff}")

print("[ INIT ] Initialization DONE!\n")