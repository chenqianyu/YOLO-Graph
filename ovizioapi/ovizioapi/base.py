import os
import win32api
from pathlib import Path
from winreg import ConnectRegistry, OpenKey, QueryValueEx, HKEY_LOCAL_MACHINE
from ovizioapi.config import settings


def get_osone_version():
    """Retrieve the OsOne version in the specified OsOne Path
    :return: OsOne version
    :rtype: str
    """
    exe_path = Path(settings.paths.osone) / "bin" / "OsOne.exe"
    props = get_file_properties(str(exe_path))
    return props["FileVersion"]


def get_vcr_versions():
    """Retrieve the present Visual C++ Redistributables on this system
    :return: Version dictionary
    :rtype: dict
    """
    path = Path(settings.paths.vcr)
    vcr_files = path.glob("msvc*.dll")

    vcr_dict = dict()
    for file in vcr_files:
        props = get_file_properties(str(file))
        vcr_dict[file.name] = {
            "Name": props["StringFileInfo"]["ProductName"],
            "Version": props["StringFileInfo"]["ProductVersion"],
        }
    return vcr_dict


def get_file_properties(fname):
    """
    Read all properties of the given file return them as a dictionary.
    :param fname: path to the file
    :type fname: str
    """
    propNames = (
        "Comments",
        "InternalName",
        "ProductName",
        "CompanyName",
        "LegalCopyright",
        "ProductVersion",
        "FileDescription",
        "LegalTrademarks",
        "PrivateBuild",
        "FileVersion",
        "OriginalFilename",
        "SpecialBuild",
    )

    props = {"FixedFileInfo": None, "StringFileInfo": None, "FileVersion": None}
    if not os.path.isfile(fname):
        raise FileNotFoundError("No such file: %s" % fname)

    try:
        # backslash as parm returns dictionary of numeric info corresponding to VS_FIXEDFILEINFO struc
        fixedInfo = win32api.GetFileVersionInfo(fname, "\\")
        props["FixedFileInfo"] = fixedInfo
        props["FileVersion"] = "%d.%d.%d.%d" % (
            fixedInfo["FileVersionMS"] / 65536,
            fixedInfo["FileVersionMS"] % 65536,
            fixedInfo["FileVersionLS"] / 65536,
            fixedInfo["FileVersionLS"] % 65536,
        )

        # \VarFileInfo\Translation returns list of available (language, codepage)
        # pairs that can be used to retreive string info. We are using only the first pair.
        lang, codepage = win32api.GetFileVersionInfo(fname, "\\VarFileInfo\\Translation")[0]

        # any other must be of the form \StringfileInfo\%04X%04X\parm_name, middle
        # two are language/codepage pair returned from above

        strInfo = {}
        for propName in propNames:
            strInfoPath = "\\StringFileInfo\\%04X%04X\\%s" % (lang, codepage, propName)
            # print str_info
            strInfo[propName] = win32api.GetFileVersionInfo(fname, strInfoPath)

        props["StringFileInfo"] = strInfo
        print("[  OK  ] File information successfully extracted!")
    except:
        print("[ ERRO ] Could not extract file information!")

    return props


def get_registry_key(hkey, keypath, property):
    """Fetches a value from the Windows registry.
    Please chose a handle key (HKEY) you want to connect to. State the
    path to the desired key and the key property you want to read.
    :param hkey: path to the file
    :type hkey: winref.HEKYType
    :param keypath: path to the key inside the handle key
    :type keypath: str
    :param property: speific name inside the key
    :type property: str
    """
    registry = ConnectRegistry(None, hkey)

    try:
        key = OpenKey(registry, keypath)
    except WindowsError:
        print(f"[ ERRO ] Registry Key {keypath} not found!")
        return None

    try:
        value = QueryValueEx(key, property)
        return value[0]
    except WindowsError:
        print(f"[ ERRO ] Key property {property} not found!")
        return None


def get_dot_net_version():
    dot_net_key = settings.registry.dotnet
    version = get_registry_key(HKEY_LOCAL_MACHINE, dot_net_key, "Version")
    release = get_registry_key(HKEY_LOCAL_MACHINE, dot_net_key, "Release")

    return f"{version}, {release}"

