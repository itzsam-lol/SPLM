import os
import ssl
import subprocess
import sys

# Monkey-patch SSL to bypass verification
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    # Older Python versions or custom builds
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

def install_requirements():
    # Use the current python executable (which should be the one in venv)
    pip_cmd = [
        sys.executable, "-m", "pip", "install", 
        "-r", "requirements.txt",
        "--prefer-binary",
        "--trusted-host", "pypi.org",
        "--trusted-host", "files.pythonhosted.org",
        "--trusted-host", "pypi.python.org",
        "--break-system-packages"
    ]
    
    print(f"Running command: {' '.join(pip_cmd)}")
    
    # Run pip as a subprocess
    # Note: Subprocesses might not inherit the monkey-patch, 
    # but setting PIP_TRUSTED_HOST and other env vars helps
    env = os.environ.copy()
    env["PYTHONHTTPSVERIFY"] = "0"
    env["PIP_TRUSTED_HOST"] = "pypi.org files.pythonhosted.org pypi.python.org"
    
    process = subprocess.Popen(
        pip_cmd, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.STDOUT, 
        text=True,
        env=env
    )
    
    for line in process.stdout:
        print(line, end="")
    
    process.wait()
    return process.returncode

if __name__ == "__main__":
    sys.exit(install_requirements())
