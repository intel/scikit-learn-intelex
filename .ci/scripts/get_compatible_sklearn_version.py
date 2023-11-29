import json
import os
import re
import subprocess
import sys

# Find current Scikit-learn major versions in release order
conda_call = subprocess.run(
    ["conda", "search", "scikit-learn", "-c", sys.argv[1]], capture_output=True
)
info = conda_call.stdout.decode("utf-8").strip()
sklearn_major_versions = list(dict.fromkeys(re.findall(r"[0-9]\.[0-9]+", info)))

# Find the location of the installed packages
conda_call = subprocess.run(["conda", "info"], capture_output=True)
info = conda_call.stdout.decode("utf-8").strip()
loc = list(re.findall(r"package\scache\s:\s[^\s]+", info))[0].split()[-1]

# Find the info.json from the Scikit-learn-intelex install
conda_call = subprocess.run(
    ["conda", "list", "scikit-learn-intelex"], capture_output=True
)
info = conda_call.stdout.decode("utf-8").strip()
pkg = re.sub(r"\s+", r"-", info.split("\n")[-1])
loc += os.sep + pkg + os.sep + "info" + os.sep + "index.json"

# Extract the min and possibly max supported major version of Scikit-learn
# from the Scikit-learn-intelex index.json information
with open(loc) as json_file:
    json_data = json.load(json_file)
    deps = [
        string for string in json_data["depends"] if string.find("scikit-learn") != -1
    ][0]

depstr = re.findall(r">=[0-9]\.[0-9]+", deps)[0][2:]
depstr2 = re.findall(r"<[0-9]\.[0-9]+", deps)


# Each Scikit-learn-intelex version supports 4 major versions
# Install the latest supported Scikit-learn for the installed scikit-learn-intelex
if depstr2:
    print(
        "scikit-learn="
        + sklearn_major_versions[max(0, sklearn_major_versions.index(depstr2) - 1)]
    )
else:
    print(
        "scikit-learn="
        + sklearn_major_versions[
            min(
                len(sklearn_major_versions) - 1,
                sklearn_major_versions.index(depstr) + 3,
            )
        ]
    )
