# Test Plate Images

Add real license plate crop images here for testing.

## Naming format
`sample_plate_<PLATE_NUMBER>.jpg`

Example: `sample_plate_gr571xc.jpg`

## How to add images
```bash
# From Home Assistant
scp root@192.168.188.20:/config/www/smart_gate/snapshot/debug/last_plate_crop.jpg \
    app/tests/fixtures/plates/sample_plate_gr571xc.jpg
```

## Privacy
Only include plate crops, no full vehicle photos.
