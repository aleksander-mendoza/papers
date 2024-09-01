import os
import platform
import shutil
import sys

PROJECT_ROOT = os.path.dirname(__file__)


def build_args(debug):
    o = platform.system()
    if o == "Windows":
        cmd = ["pwsh.exe", "-c"]
    else:
        cmd = ["/bin/bash", "-c"]
    shell_steps = ["cd ecc_py"]
    cargo_profile = 'debug' if debug else 'release'
    maturin = shutil.which("maturin")
    if maturin is None:
        print("Maturin not installed")
        exit(1)
    shell_steps.append(maturin + " build -i \"" + sys.executable + "\" --" + cargo_profile)
    shell_steps.append("cd ..")
    WHEELS_DIR = "target/wheels/"
    pip_path = shutil.which("pip")
    if o == "Windows":
        shell_steps.append(pip_path + f" install --force-reinstall \"$(Get-ChildItem -Path {WHEELS_DIR} | Sort-Object LastWriteTime -Descending | Select-Object -first 1 )\"")
    else:
        shell_steps.append(pip_path + f" install --force-reinstall \"{WHEELS_DIR}$(ls -1t {WHEELS_DIR} | head -1)\"")
    cmd.append(" && ".join(shell_steps))
    return cmd


if __name__ == '__main__':
    import argparse
    import subprocess

    args = argparse.ArgumentParser("Building tool")
    args.add_argument("-d", "--debug", default=False, action='store_true')
    args = args.parse_args()
    exit(subprocess.call(build_args(args.debug)))
