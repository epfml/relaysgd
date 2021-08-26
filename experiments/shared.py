from train import config
from jobmonitor.api import (
    kubernetes_schedule_job,
    register_job,
    upload_code_package as ucp,
)
from jobmonitor.connections import mongo

excluded_files = [
    "core",
    "output.tmp",
    ".vscode",
    ".git",
    "*.pyc",
    "._*",
    "data",
    "*.exr",
    "__pycache__",
    ".ipynb_checkpoints",
    "jupyter",
    ".pylintrc",
    ".gitignore",
    ".AppleDouble",
]

def upload_code_package():
    code_package, files_uploaded = ucp(".", excludes=excluded_files)
    print("Uploaded {} files.".format(len(files_uploaded)))
    return code_package

r = []

def remote_exec(statement):
    print(statement)
    # subprocess.check_call(
    #     ["ssh", "-o", "StrictHostKeyChecking=no", "cluster.europe-west1-b.rank1-gradient-compression", "--"] + statement.split(" ")
    # )