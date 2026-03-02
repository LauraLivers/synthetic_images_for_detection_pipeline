## Repo
Multi Root Workspace

### Python Version Management with uv

This workspace contains multiple projects with different Python version requirements:
- **DiffCL**: Python 3.10
- **MobI**: Python 3.8

#### Initial Setup

1. **Install uv** (if not already installed):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Install required Python versions**:
   ```bash
   uv python install 3.10
   uv python install 3.8
   ```

3. **Verify installations**:
   ```bash
   uv python list
   ```

#### Using the Correct Python Version

Each project has a `.python-version` file that automatically tells `uv` which Python version to use.

**When working in DiffCL:**
```bash
cd DiffCL
uv python pin 3.10  # This is already done via .python-version
uv venv              # Create virtual environment with Python 3.10
source .venv/bin/activate  # Activate environment
uv pip install -r requirements.txt
```

**When working in MobI:**
```bash
cd MobI
uv python pin 3.8   # This is already done via .python-version
uv venv             # Create virtual environment with Python 3.8
source .venv/bin/activate  # Activate environment
uv pip install -r requirements-linux-cuda.txt
```

#### Switching Between Projects

The `.python-version` file ensures the correct Python version is used automatically:

```bash
# Switch to DiffCL
cd ../DiffCL
uv venv  # Automatically uses Python 3.10
source .venv/bin/activate

# Switch to MobI
cd ../MobI
uv venv  # Automatically uses Python 3.8
source .venv/bin/activate
```

#### Running Scripts with uv

You can also run scripts directly without activating the virtual environment:

```bash
# In DiffCL
cd DiffCL
uv run python script.py  # Uses Python 3.10

# In MobI
cd MobI
uv run python script.py  # Uses Python 3.8
```

#### VS Code Configuration

VS Code should automatically detect the correct Python interpreter from each project's `.venv` folder. If not:

1. Open Command Palette (Ctrl+Shift+P)
2. Type "Python: Select Interpreter"
3. Choose the interpreter from the respective `.venv/bin/python` for each folder

#### Troubleshooting

- **Check current Python version**: `uv run python --version`
- **List installed Python versions**: `uv python list`
- **Reinstall venv if needed**: Remove `.venv` folder and run `uv venv` again
- **Check which Python uv will use**: `uv python find` (in project directory)

### New Folders
1. physically create new folder
2. add to `myworkspace.code-workspace`

### Report

