<img src="./logo.png" alt="Smart Gate" />

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

### 4. Testing

#### Run All Tests

Execute the complete test suite:
```bash
pytest -v
```

#### Run Specific Test File

Test a single module:
```bash
# OCR comparison tests (old vs AI SR)
pytest app/tests/test_enhancement_ocr_comparison.py -v

# Image processing tests
pytest app/tests/test_image_processing.py -v

# OCR pattern extraction tests
pytest app/tests/test_ocr.py -v
```

#### Run Single Test Function

Execute a specific test with detailed output:
```bash
pytest app/tests/test_enhancement_ocr_comparison.py::test_ocr_comparison -v -s
```

**Flags:**
- `-v` (verbose): Show detailed test names
- `-s`: Show print statements and logging output

#### Run Tests with Coverage

Generate coverage report:
```bash
# Terminal output
pytest --cov=. --cov-report=term-missing

# HTML report
pytest --cov=. --cov-report=html
# Open htmlcov/index.html in browser
```

#### Run Tests Matching Pattern

Execute tests by name pattern:
```bash
# Run all OCR-related tests
pytest -k "ocr" -v

# Run all comparison tests
pytest -k "comparison" -v
```

#### Quick Test Commands
```bash
# Fast: skip slow tests
pytest -v -m "not slow"

# Debug: stop on first failure
pytest -x

# Watch mode: re-run on file changes (requires pytest-watch)
ptw -- -v
```

#### Add Test Images

For OCR comparison tests, add real plate images:
```bash
# Copy from Home Assistant
scp root@HA_IP:/config/www/smart_gate/snapshot/debug/last_plate_crop.jpg \
    app/tests/fixtures/plates/sample_plate_GR571XC.png

# Run comparison
pytest app/tests/test_enhancement_ocr_comparison.py::test_ocr_comparison_summary -v -s
```

**Expected output:**
```
======================================================================
Image: sample_plate_GR571XC.png
======================================================================
Expected plate:  GR571XC
Old algorithm:   'CR571XC' (confidence: 0.810)
New AI SR:       'GR571XC' (confidence: 0.967)

Old match: ❌
New match: ✅ IMPROVED
```

## Git Hooks

Pre-commit hook runs all tests before allowing a commit.

To skip (not recommended):
```bash
git commit --no-verify
```





