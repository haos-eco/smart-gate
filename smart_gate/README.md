<img src="https://raw.githubusercontent.com/haos-eco/smart-gate/main/smart_gate/logo.png" alt="Smart Gate" />

# Smart Gate - License Plate Recognition

Automatic gate AI powered opener using YOLO + EasyOCR for license plate recognition.

**Author**: [haz3](https://github.com/andreaemmanuele)

![Supports aarch64 Architecture][aarch64-shield]
![Supports amd64 Architecture][amd64-shield]

[aarch64-shield]: https://img.shields.io/badge/aarch64-yes-green.svg
[amd64-shield]: https://img.shields.io/badge/amd64-yes-green.svg

## Features
- 🚗 Automatic license plate detection using YOLOv11
- 🔍 OCR with EasyOCR for Italian plates (AA123AA format)
- 🤖 AI Super-Resolution (EDSR 2x) for enhanced accuracy on low-res crops
- 🌙 Automatic IR detection and exposure correction for night shots
- 🎯 Multi-attempt recognition — best result selected by combined YOLO + OCR score
- 🔎 Fuzzy matching for OCR errors (up to 2 character tolerance)
- 📍 GPS-based security — fuzzy matches require the plate owner to be home
- 🔒 Per-plate person entity — each plate linked to its specific owner
- 📊 Debug mode with per-attempt snapshot history
