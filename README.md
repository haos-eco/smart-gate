## Development Setup

### 1. Clone repository
```bash
git clone https://github.com/andreaemmanuele/haos_smartgate.git
cd haos_smartgate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
pip install -r requirements-test.txt
```

### 3. Setup Git hooks (IMPORTANT!)
```bash
./setup-hooks.sh
```

This configures pre-commit hooks to run tests automatically before each commit.

### 4. Run tests
```bash
pytest -v
```

## Git Hooks

Pre-commit hook runs all tests before allowing a commit.

To skip (not recommended):
```bash
git commit --no-verify
```
