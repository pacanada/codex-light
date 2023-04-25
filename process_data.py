import git
from tqdm import tqdm
from utils import get_root, save_train_test_set
from pathlib import Path
LIST_REPOS = ["https://github.com/TheAlgorithms/Python.git"]
REPOS_DIR = get_root() / "data" / "repos_python"

def clone_repos():
    for repo in LIST_REPOS:
        print(f"Cloning {repo}")
        git.Git(REPOS_DIR).clone(repo)

def parse_python_files():
    python_files = list(Path(REPOS_DIR).rglob("*.py"))

    data = ""
    for python_file in tqdm(python_files):
        with open(python_file, "r") as f:
            data = data + "\n" + f.read()
    with open(get_root() / "data" / "python_text.txt", "w") as f:
        f.write(data)

    print("Chars:\n", list(set(data)), len(set(data)))



# clone
#clone_repos()
# parsing files
#parse_python_files()
save_train_test_set()


