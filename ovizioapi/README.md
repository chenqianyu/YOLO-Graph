# OvizioApi

API for accessing Ovizio Reconstruction Tools.

## Installation
:warning: Due to some .NET dependencies, this package only works on a Windows machine or inside a virtual box running Windows!

:warning: Due to limitations of [Pythonnet](https://github.com/pythonnet/pythonnet) we currently only support **Python 3.8**

### OsOne
Make sure OsOne is installed on your machine. You can download it [here](https://deepsea.ldv.ei.tum.de/f/c775769f697f49c184e8/?dl=1). Restart your computer after installation.


### Python Repo
- Clone the reposiory to your local disk.
```bash
git clone ...
```
- Check if you have the correct python version
```bash
python --version
```
- Create a virtual python environment named _ovapi_
```bash
python -m venv ovapi
```
- Copy the _pip.ini_ file into the newly created _ovapi_ folder
```bash
cp pip.ini ovapi/
```
- Activate the virtual environment (depending if you are on CMD or Powershell)
```bash
ovapi\Scripts\activate.bat|Activate.ps1
```
- Navigate in the folder which holds the `setup.py` and install the package using pip:
```bash
(ovapi)$ pip install .[examples]
```

## Usage
- Make sure your virtual environment is activated. You will see _(ovapi)_ in front you your command prompt.
```bash
$ ovapi\Scripts\activate.bat|Activate.ps1`
(ovapi)$
```
- Start the jupyter notebook server:
```bash
(ovapi)$ jupyter notebook
```
- Open the notebook _notebooks\containerize.ipynb_
- Follow the instructions in the notebook.
- Make sure to fill out the meta data dictionary correctly!


## Low Level Functions
Reconstructing the **Phase** images works like this. **Intensity** (Amplitude) and **Hologram** are analogue.
```python
from ovizioapi.capture import OvizioCapture

# Create a capture object
cpt = OvizioCapture("path/to/your/Capture 1.h5")

# Reconstruct the phase images of this capture
for j in range(len(cpt)):
    phase_image = cpt.get_phase(j)
    # ... store the image some where

```
You can access the metadata like this:
```python
from ovizioapi.capture import OvizioCapture

# Create a capture object
cpt = OvizioCapture("path/to/your/Capture 1.h5")
cpt.metadata # {"height": 1536, "width": 2048, "count": 10, "pixel_width": 3.45, "wave_length": 530.0, "magnification": 40.0}
```
Hint :bulb:
> Checkout the [example notebook](notebooks/export_images.ipynb) how to read, plot and export images.

## Errors
Machines which do not have a complete installation of OsOne (normally every PC which is not connected to a microscope) misses some personal config files. This does **not affect the integrity of the reconstruction**. Therefore, the following messages can be **ignored**:

- log4net:ERROR
```
log4net:ERROR XmlConfigurator: Failed to find configuration section 'log4net' in the application's .config file. Check your .config file for the <log4net> and <configSections> elements. The configuration section should look like: <section name="log4net" type="log4net.Config.Log4NetConfigurationSectionHandler,log4net" />
```

## FAQs

### Error: win32api: DLL load failed while importing win32api:
```
ImportError                               Traceback (most recent call last)
Input In [4], in <module>
----> 1 from ovizioapi.capture import OvizioCapture

File \\nas.ads.mwn.de\ga68bow\tum-pc\dokumente\git\ovizioapi\ovizioapi\__init__.py:13, in <module>
     10 print("[ INFO ] Using OsOne Path: %s" % cfg["OSONEPATH"])
     12 if os.name == "nt":
---> 13     from .utils import get_osone_version, get_dot_net_version, get_vcr_versions
     15     print("[ INFO ] Checking OsOne Version")
     16     version = get_osone_version()

File \\nas.ads.mwn.de\ga68bow\tum-pc\dokumente\git\ovizioapi\ovizioapi\utils.py:2, in <module>
      1 import os
----> 2 import win32api
      3 from pathlib import Path
      4 from winreg import ConnectRegistry, OpenKey, QueryValueEx, HKEY_LOCAL_MACHINE

ImportError: DLL load failed while importing win32api: Die angegebene Prozedur wurde nicht gefunden.
```
### Solution:
This may happen if you use pip in conda environment or vice versa. Try to reinstall pywin32 in your environment.
```
pip install pywin32
```
or
```
conda install pywin32
```

### Error: Unable to find assembly 
It might happen that windows block unknown DLLs. You have to allow the DLLs _OvizioApiNet.dll_ and _OvizioCoreWrapper.dll_. 

```
FileNotFoundException: Unable to find assembly 'OvizioApiNet'.
   bei Python.Runtime.CLRModule.AddReference(String name)
```
### Solution
- See instructions above -> Oviziop API Binaries+
- Make sure the Visual C++ Redistributables are installed correctly on your machine

## Tests
To make sure the .NET components where installed correctly you can run some tests. Even if the code does **not crash or output an error** the reconstruction algorithm might output a matrix of plain `0` or `NAN`. This is obviously not correct!

You can perform the following steps to make sure the reconstruction works flawlessly:
- Make sure the **virtual environment** you installed ovozioapi in is actiavted. You will see the environmentname with brackets in front of your commandline.
```
(env)$ ...
```
- Install `pytest`
```bash
(env)$ pip install pytest
```
- Navigate to the main folder (the one which contains the setup.py)
- Start the tests using `pytest`
```
(env)$ pytest tests
```
- Check to output for failed tests (indicated with an `F`).
- If no `F`s appear the API is working correctly. Congratulations!
