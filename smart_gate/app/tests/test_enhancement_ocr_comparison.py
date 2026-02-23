import os
import sys
import pytest
import cv2
import numpy as np
import re

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from image_processing import crop_white_area, preprocess_plate
from ocr import extract_plate_pattern, fix_common_ocr_errors, ocr_plate
import easyocr

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), 'fixtures', 'plates')

@pytest.fixture(scope="session", autouse=True)
def preload_trocr():
    from trocr import load_trocr
    load_trocr()

def get_test_images():
    """Get all test images with expected plate numbers from filename"""
    if not os.path.exists(FIXTURES_DIR):
        return []

    images = []
    for filename in os.listdir(FIXTURES_DIR):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            # Format: sample_plate_GR571XC.jpg -> GR571XC
            match = re.search(r'plate_([A-Z0-9]+)', filename)
            expected_plate = match.group(1).upper() if match else None
            images.append({
                'path': os.path.join(FIXTURES_DIR, filename),
                'filename': filename,
                'expected': expected_plate
            })

    return images

def test_fixtures_exist():
    """Verify that test fixtures directory exists and contains images"""
    assert os.path.exists(FIXTURES_DIR), \
        f"Fixtures directory not found: {FIXTURES_DIR}"

    images = get_test_images()

    if not images:
        pytest.skip(
            f"No test images found in {FIXTURES_DIR}. "
            "Please add real plate crop images for testing.\n"
            "Example: scp root@HA_IP:/config/www/smart_gate/snapshot/debug/last_plate_crop.jpg "
            f"{FIXTURES_DIR}/sample_plate_GR571XC.jpg"
        )

    print(f"\n✅ Found {len(images)} test image(s):")
    for img_info in images:
        size = os.path.getsize(img_info['path']) / 1024
        expected = img_info['expected'] or 'UNKNOWN'
        print(f"   - {img_info['filename']} ({size:.1f} KB) - Expected: {expected}")

@pytest.fixture(scope="module")
def ocr_reader():
    """Load EasyOCR reader once for all tests"""
    print("\nLoading EasyOCR reader...")
    return easyocr.Reader(['en'], gpu=False)

def ocr_with_old_algorithm(reader, img):
    """OCR with OLD algorithm: simple upscale + sharpening kernel, no preprocess_plate"""
    img = crop_white_area(img)
    h, w = img.shape[:2]

    # Old: simple upscale to 400px
    if h < 400:
        scale = 400 / h
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4)

    # Old: 3x3 sharpening kernel
    kernel_sharpen = np.array([[-1, -1, -1],
                               [-1,  9, -1],
                               [-1, -1, -1]])
    img = cv2.filter2D(img, -1, kernel_sharpen)

    results = reader.readtext(img, allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')

    if not results:
        return None, 0.0

    # Get best result
    valid_results = [(re.sub(r"[^A-Z0-9]", "", text.upper()), conf)
                     for _, text, conf in results if conf > 0.5]

    if not valid_results:
        return None, 0.0

    valid_results.sort(key=lambda x: len(x[0]), reverse=True)
    best, conf = valid_results[0]
    best = fix_common_ocr_errors(extract_plate_pattern(best))
    return best, conf


def ocr_with_new_algorithm(reader, img):
    """OCR with NEW algorithm: full preprocess_plate pipeline"""
    plate, conf = ocr_plate(reader, img.copy(), debug=False)
    return plate if plate else None, conf


@pytest.mark.parametrize("test_image", get_test_images())
def test_ocr_comparison(ocr_reader, test_image):
    """Compare OCR results between old and new pipeline"""
    img = cv2.imread(test_image['path'])
    assert img is not None, f"Failed to load {test_image['path']}"

    old_result, old_conf = ocr_with_old_algorithm(ocr_reader, img.copy())
    new_result, new_conf = ocr_with_new_algorithm(ocr_reader, img.copy())

    print(f"\n{'='*70}")
    print(f"Image: {test_image['filename']}")
    print(f"{'='*70}")
    print(f"Expected:      {test_image['expected'] or 'UNKNOWN'}")
    print(f"Old algorithm: '{old_result or 'N/A'}' (confidence: {old_conf:.3f})")
    print(f"New pipeline:  '{new_result or 'N/A'}' (confidence: {new_conf:.3f})")

    if test_image['expected']:
        old_match = old_result == test_image['expected']
        new_match = new_result == test_image['expected']

        print(f"\nOld match: {'✅' if old_match else '❌'}")
        print(f"New match: {'✅' if new_match else '❌'}")

        if old_match:
            assert new_match, \
                f"New pipeline regressed: '{new_result}' vs expected '{test_image['expected']}'"

    assert old_result is not None or new_result is not None, \
        "Both algorithms failed to detect plate"


def test_ocr_comparison_summary(ocr_reader):
    """Summary report of all OCR comparisons"""
    test_images = get_test_images()

    if not test_images:
        pytest.skip("No test images available")

    results = []

    print("\n🔍 Running OCR comparison on all test images...")

    for test_image in test_images:
        img = cv2.imread(test_image['path'])
        if img is None:
            continue

        old_result, old_conf = ocr_with_old_algorithm(ocr_reader, img.copy())
        new_result, new_conf = ocr_with_new_algorithm(ocr_reader, img.copy())

        results.append({
            'filename': test_image['filename'],
            'expected': test_image['expected'],
            'old_result': old_result,
            'old_conf': old_conf,
            'new_result': new_result,
            'new_conf': new_conf,
            'old_match': old_result == test_image['expected'] if test_image['expected'] else None,
            'new_match': new_result == test_image['expected'] if test_image['expected'] else None,
        })

    print("\n" + "="*100)
    print("OCR COMPARISON SUMMARY — OLD vs NEW PIPELINE (preprocess_plate)")
    print("="*100)
    print(f"{'Image':<30} {'Expected':<10} {'Old Result':<20} {'New Result':<20} {'Status':<20}")
    print("-"*100)

    old_correct = new_correct = total_with_expected = improvements = regressions = 0

    for r in results:
        status = ""
        if r['expected']:
            total_with_expected += 1
            if r['old_match']:
                old_correct += 1
            if r['new_match']:
                new_correct += 1

            if r['new_match'] and not r['old_match']:
                status = "✅ IMPROVED"
                improvements += 1
            elif not r['new_match'] and r['old_match']:
                status = "❌ REGRESSED"
                regressions += 1
            elif r['new_match'] and r['old_match']:
                status = "✓ Both OK (↑ conf)" if r['new_conf'] > r['old_conf'] + 0.05 else "✓ Both OK"
            else:
                status = "✗ Both wrong"

        old_display = f"{r['old_result'] or 'N/A'} ({r['old_conf']:.2f})"
        new_display = f"{r['new_result'] or 'N/A'} ({r['new_conf']:.2f})"

        print(f"{r['filename']:<30} {r['expected'] or 'N/A':<10} {old_display:<20} {new_display:<20} {status:<20}")

    print("-"*100)

    if total_with_expected > 0:
        old_accuracy = (old_correct / total_with_expected) * 100
        new_accuracy = (new_correct / total_with_expected) * 100

        print(f"\n📊 STATISTICS:")
        print(f"   Old algorithm accuracy:  {old_correct}/{total_with_expected} ({old_accuracy:.1f}%)")
        print(f"   New pipeline accuracy:   {new_correct}/{total_with_expected} ({new_accuracy:.1f}%)")
        print(f"   Improvement:             {new_accuracy - old_accuracy:+.1f}%")
        print(f"   Improved plates:         {improvements}")
        print(f"   Regressed plates:        {regressions}")

        assert new_accuracy >= old_accuracy, \
            f"New pipeline regressed by {old_accuracy - new_accuracy:.1f}%"

        if improvements > 0:
            print(f"\n✅ New pipeline improved {improvements} plate(s)!")

    print("="*100)

@pytest.mark.skipif(
    len(get_test_images()) == 0,
    reason="No test images available"
)
def test_new_pipeline_never_worse(ocr_reader):
    """Ensure new pipeline never performs worse than old algorithm"""
    regressions = []

    for test_image in get_test_images():
        if not test_image['expected']:
            continue

        img = cv2.imread(test_image['path'])
        if img is None:
            continue

        old_result, _ = ocr_with_old_algorithm(ocr_reader, img.copy())
        new_result, _ = ocr_with_new_algorithm(ocr_reader, img.copy())

        if old_result == test_image['expected'] and new_result != test_image['expected']:
            regressions.append({
                'image': test_image['filename'],
                'expected': test_image['expected'],
                'old': old_result,
                'new': new_result,
            })

    if regressions:
        msg = "\n❌ New pipeline regressions detected:\n"
        for r in regressions:
            msg += f"  {r['image']}: expected '{r['expected']}', old='{r['old']}', new='{r['new']}'\n"
        pytest.fail(msg)


def test_sr_model_loads():
    """Test that AI super-resolution model loads correctly"""
    from image_processing import get_sr_model
    sr = get_sr_model()
    if sr is None:
        print("\n⚠️  AI SR model not available, bicubic fallback will be used")
    else:
        print("\n✅ AI SR model loaded successfully")
