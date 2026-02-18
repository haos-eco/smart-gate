import os
import sys
import pytest
import cv2
import numpy as np
import re

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from image_processing import crop_white_area, enhance_plate_ai_sr
from ocr import extract_plate_pattern, fix_common_ocr_errors
import easyocr

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), 'fixtures', 'plates')

def get_test_images():
    """Get all test images with expected plate numbers from filename"""
    if not os.path.exists(FIXTURES_DIR):
        return []

    images = []
    for filename in os.listdir(FIXTURES_DIR):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            # Extract expected plate from filename
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
        size = os.path.getsize(img_info['path']) / 1024  # KB
        expected = img_info['expected'] or 'UNKNOWN'
        print(f"   - {img_info['filename']} ({size:.1f} KB) - Expected: {expected}")

@pytest.fixture(scope="module")
def ocr_reader():
    """Load EasyOCR reader once for all tests"""
    print("\nLoading EasyOCR reader...")
    return easyocr.Reader(['en'], gpu=False)

def ocr_with_old_algorithm(reader, img):
    """OCR with OLD algorithm (simple upscale to 400px, no AI)"""
    img = crop_white_area(img)
    h, w = img.shape[:2]

    # OLD: Simple upscale to 400px
    target_h = 400
    if h < target_h:
        scale = target_h / h
        img = cv2.resize(img, None, fx=scale, fy=scale,
                         interpolation=cv2.INTER_LANCZOS4)

    # OLD: Simple sharpening kernel
    kernel_sharpen = np.array([[-1,-1,-1],
                               [-1, 9,-1],
                               [-1,-1,-1]])
    img = cv2.filter2D(img, -1, kernel_sharpen)

    # OCR
    results = reader.readtext(img, allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')

    if not results:
        return None, 0.0

    # Get best result
    valid_results = [(re.sub(r"[^A-Z0-9]", "", text.upper()), conf)
                     for _, text, conf in results if conf > 0.5]

    if not valid_results:
        return None, 0.0

    valid_results.sort(key=lambda x: len(x[0]), reverse=True)
    best = valid_results[0][0]
    conf = valid_results[0][1]

    # Apply fixes
    best_extracted = extract_plate_pattern(best)
    best_fixed = fix_common_ocr_errors(best_extracted)

    return best_fixed, conf

def ocr_with_new_algorithm(reader, img):
    """OCR with NEW algorithm (AI super-resolution)"""
    from ocr import ocr_plate

    # Get result with AI SR
    result = ocr_plate(reader, img.copy(), debug=False)

    # Get confidence by running OCR again
    img_processed = crop_white_area(img)
    img_processed = enhance_plate_ai_sr(img_processed, debug=False)

    results = reader.readtext(img_processed,
                              allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')

    conf = results[0][2] if results else 0.0

    return result if result else None, conf

@pytest.mark.parametrize("test_image", get_test_images())
def test_ocr_comparison(ocr_reader, test_image):
    """Compare OCR results between old and AI super-resolution algorithms"""
    img = cv2.imread(test_image['path'])
    assert img is not None, f"Failed to load {test_image['path']}"

    # Run both algorithms
    old_result, old_conf = ocr_with_old_algorithm(ocr_reader, img.copy())
    new_result, new_conf = ocr_with_new_algorithm(ocr_reader, img.copy())

    # Log results
    print(f"\n{'='*70}")
    print(f"Image: {test_image['filename']}")
    print(f"{'='*70}")
    print(f"Expected plate:  {test_image['expected'] or 'UNKNOWN'}")
    print(f"Old algorithm:   '{old_result or 'N/A'}' (confidence: {old_conf:.3f})")
    print(f"New AI SR:       '{new_result or 'N/A'}' (confidence: {new_conf:.3f})")

    # Compare with expected if available
    if test_image['expected']:
        old_match = old_result == test_image['expected']
        new_match = new_result == test_image['expected']

        print(f"\nOld match: {'✅' if old_match else '❌'}")
        print(f"New match: {'✅' if new_match else '❌'}")

        # New algorithm should be at least as good as old
        if old_match:
            assert new_match, \
                f"AI SR regressed: '{new_result}' vs expected '{test_image['expected']}'"

    # Both should return something
    assert old_result is not None or new_result is not None, \
        "Both algorithms failed to detect plate"

def test_ocr_comparison_summary(ocr_reader, capsys):
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
            'new_match': new_result == test_image['expected'] if test_image['expected'] else None
        })

    # Print summary table
    print("\n" + "="*100)
    print("OCR COMPARISON SUMMARY - OLD vs AI SUPER-RESOLUTION")
    print("="*100)
    print(f"{'Image':<30} {'Expected':<10} {'Old Result':<15} {'AI SR Result':<15} {'Status':<20}")
    print("-"*100)

    old_correct = 0
    new_correct = 0
    total_with_expected = 0
    improvements = 0
    regressions = 0

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
                # Check if confidence improved
                if r['new_conf'] > r['old_conf'] + 0.05:
                    status = "✓ Both OK (↑ conf)"
                else:
                    status = "✓ Both OK"
            else:
                status = "✗ Both wrong"

        old_display = f"{r['old_result'] or 'N/A'} ({r['old_conf']:.2f})"
        new_display = f"{r['new_result'] or 'N/A'} ({r['new_conf']:.2f})"

        print(f"{r['filename']:<30} {r['expected'] or 'N/A':<10} {old_display:<15} {new_display:<15} {status:<20}")

    print("-"*100)

    if total_with_expected > 0:
        old_accuracy = (old_correct / total_with_expected) * 100
        new_accuracy = (new_correct / total_with_expected) * 100

        print(f"\n📊 STATISTICS:")
        print(f"   Old algorithm accuracy: {old_correct}/{total_with_expected} ({old_accuracy:.1f}%)")
        print(f"   AI SR accuracy:         {new_correct}/{total_with_expected} ({new_accuracy:.1f}%)")
        print(f"   Improvement:            {new_accuracy - old_accuracy:+.1f}%")
        print(f"   Improved plates:        {improvements}")
        print(f"   Regressed plates:       {regressions}")

        # Assert that new algorithm doesn't regress
        assert new_accuracy >= old_accuracy, \
            f"AI SR regressed by {old_accuracy - new_accuracy:.1f}%"

        if improvements > 0:
            print(f"\n✅ AI Super-Resolution improved {improvements} plate(s)!")

    print("="*100)

@pytest.mark.skipif(
    len(get_test_images()) == 0,
    reason="No test images available"
)
def test_ai_sr_never_makes_worse(ocr_reader):
    """Ensure AI SR never performs worse than old algorithm on any image"""
    test_images = get_test_images()

    regressions = []

    for test_image in test_images:
        if not test_image['expected']:
            continue

        img = cv2.imread(test_image['path'])
        if img is None:
            continue

        old_result, _ = ocr_with_old_algorithm(ocr_reader, img.copy())
        new_result, _ = ocr_with_new_algorithm(ocr_reader, img.copy())

        old_correct = old_result == test_image['expected']
        new_correct = new_result == test_image['expected']

        if old_correct and not new_correct:
            regressions.append({
                'image': test_image['filename'],
                'expected': test_image['expected'],
                'old': old_result,
                'new': new_result
            })

    if regressions:
        msg = "\n❌ AI SR regressions detected:\n"
        for r in regressions:
            msg += f"  {r['image']}: expected '{r['expected']}', old got '{r['old']}', AI SR got '{r['new']}'\n"
        pytest.fail(msg)

def test_ai_sr_model_loads():
    """Test that AI super-resolution model loads correctly"""
    from image_processing import get_sr_model

    # Try to get model
    sr = get_sr_model()

    # Model should either load or gracefully fall back
    # (we don't fail if model unavailable, just log it)
    if sr is None:
        print("\n⚠️  AI SR model not available, will use fallback upscaling")
    else:
        print("\n✅ AI SR model loaded successfully")
