from pathlib import Path
from dynaconf import Dynaconf, Validator
from xmltodict import parse

def read_xml_file(file_path, new_root: str = None):
    with open(file_path, "r") as f:
        data = parse(f.read())
    if new_root is not None:
        return data[new_root]
    return data


CONFIG_FOLDER = Path(__file__).parent


settings = Dynaconf(
    environments=True,
    load_dotenv=True,
    envvar_prefix="OVIZIOAPI",
    settings_files=CONFIG_FOLDER / "settings.toml",
    validators=[
        Validator("paths.vcr", "paths.osone", "registry.dotnet", must_exist=True),
    ],
)

# `envvar_prefix` = export envvars with `export OVIZIOAPI_FOO=bar`.
# `settings_files` = Load these files in the order.

settings.validators.validate()

default_config_path = CONFIG_FOLDER / "DefaultReconstructionParameters.xml"
default_reconstruction_config = read_xml_file(default_config_path, "ovizio")