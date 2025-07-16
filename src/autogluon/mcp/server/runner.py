def main():
    import os
    import subprocess

    subprocess.run(["/bin/bash", os.path.join(os.path.dirname(__file__), "start_services.sh")])
