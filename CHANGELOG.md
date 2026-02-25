# Changelog

All notable changes to this project will be documented in this file.

## [unreleased]

### 🐛 Bug Fixes

- Removed duplicated import from main.py

### 📚 Documentation

- Update changelog [skip ci]
- Update changelog [skip ci]

### ⚙️ Miscellaneous Tasks

- Fix readme + update version in repository.json

## [2.0.0] - 2026-02-23

### 🐛 Bug Fixes

- Add missing dependencies in dockerfile

### 📚 Documentation

- Update changelog [skip ci]

### ⚙️ Miscellaneous Tasks

- Bump to major version 2.0.0

## [1.2.5] - 2026-02-23

### 📚 Documentation

- Update changelog [skip ci]

## [1.2.4] - 2026-02-23

### 🚀 Features

- Early exit when the plate is an exact match, write log using a different thread

### 🐛 Bug Fixes

- Dashboard view
- Logs snapshot, bump version to 1.2.4

### 🚜 Refactor

- Dashboard view
- Dashboard ui

### 📚 Documentation

- Update changelog [skip ci]
- Update changelog [skip ci]
- Update changelog [skip ci]

### ⚙️ Miscellaneous Tasks

- Bump version to 1.2.5

## [1.2.3] - 2026-02-21

### 🚀 Features

- Track access logs + ha dashboard with access logs

### 🐛 Bug Fixes

- Main variable error
- Exact plate match doesn't require score
- Vehicle detection, smart notification long press shows snapshot and now live camera now
- Remove extra useless camera snapshot
- Headlight overexposure

### 📚 Documentation

- Update changelog [skip ci]
- Update changelog [skip ci]
- Update changelog [skip ci]
- Update changelog [skip ci]
- Update changelog [skip ci]
- Update changelog [skip ci]

### 🧪 Testing

- Adds notifications tests
- Fix hass url

### ⚙️ Miscellaneous Tasks

- Fix config.yaml
- Bump version to 1.2.3

## [1.2.2] - 2026-02-20

### 🚀 Features

- Notify allowed devices if a car stops forward the gate

### 🐛 Bug Fixes

- Prefer plates that fully matches allowed plates
- Test_rot_snapshot paths
- Retrieving notify_services options

### 🚜 Refactor

- Moved all notification functions into dedicated file, add costants.py

### 📚 Documentation

- Update changelog [skip ci]
- Update DOCS.md and README.md
- Update changelog [skip ci]
- Update DOCS.md and README.md
- Update changelog [skip ci]
- Update changelog [skip ci]

### 🧪 Testing

- Add test_roi_snapshot

## [1.2.1] - 2026-02-19

### 🚀 Features

- Update gh action changelog.yaml to release new version

### 🐛 Bug Fixes

- Ocr tests

### 🚜 Refactor

- Update changelog path to root folder

### 📚 Documentation

- Update changelog [skip ci]
- Update changelog [skip ci]

## [1.2.0] - 2026-02-19

### 🚀 Features

- Implement fuzzy matching and person entity for allowed plates

### 🚜 Refactor

- Trying different approach to determine allowed plates

### 📚 Documentation

- Update changelog [skip ci]
- Update changelog [skip ci]
- Update changelog [skip ci]

### ⚙️ Miscellaneous Tasks

- Update README
- Bump version to 1.2.0

## [1.1.2] - 2026-02-18

### 🚀 Features

- *(dockerfile)* Enhancements

### 🐛 Bug Fixes

- *(dockerfile)* Dependencies versions
- Bump version to 1.1.2

### 📚 Documentation

- Update changelog [skip ci]
- Update changelog [skip ci]

## [1.1.1] - 2026-02-18

### 🐛 Bug Fixes

- Logo in readme and docs
- Github action update changelog
- Logo in readme and docs
- App readme

### 🚜 Refactor

- App structure

### 📚 Documentation

- Update changelog [skip ci]
- Update changelog [skip ci]
- Update changelog [skip ci]

### ⚙️ Miscellaneous Tasks

- Update opencv-contrib-python to 4.13.0.92 (latest stable)

## [1.1.0] - 2026-02-18

### 🚀 Features

- Enhance plate resolution using ai model

### 🐛 Bug Fixes

- Model path, readme
- Removing destructive post processing

### 📚 Documentation

- Update changelog [skip ci]
- Update changelog [skip ci]

### ⚙️ Miscellaneous Tasks

- Bump to major version 1.1.0, add DOCS, update README

## [.1.0.6] - 2026-02-18

### 🐛 Bug Fixes

- Resize to standard ha app dimensions icon and logo.png

### 🚜 Refactor

- Split main.py in multiple files, added tests
- Added .githooks precommit for automated test

### 📚 Documentation

- Update changelog [skip ci]
- Update changelog [skip ci]

### ⚙️ Miscellaneous Tasks

- Bump version to 1.0.6, update README

## [1.0.5] - 2026-02-17

### 🚀 Features

- Trying to resolve sovraexposure for plates

### 🐛 Bug Fixes

- Enhance plate reading, add fix sovraexposure for night mode

### 📚 Documentation

- Update changelog [skip ci]
- Update changelog [skip ci]
- Update changelog [skip ci]
- Update changelog [skip ci]
- Update changelog [skip ci]

### ⚙️ Miscellaneous Tasks

- Update README
- Bump version to 1.0.5

## [1.0.4] - 2026-02-17

### 🐛 Bug Fixes

- *(config.yaml)* Paths
- Added debug_path in config.yaml and main.py, removed useless comments

### 📚 Documentation

- Update changelog [skip ci]
- Update changelog [skip ci]

### ⚙️ Miscellaneous Tasks

- *(README)* Updated references to github repo, added author
- Bump version to 1.0.4
- Update repository.json

## [1.0.3] - 2026-02-17

### 📚 Documentation

- Update changelog [skip ci]
- Update changelog [skip ci]
- Update changelog [skip ci]
- Update changelog [skip ci]

### ⚙️ Miscellaneous Tasks

- Delete build.yaml, add git repo url to config.yaml
- Remove subfolder smart_gate
- Add attributions in README, add GNU LICENSE
- Bump version to 1.0.3
- *(config.yaml)* Updates git repo in url

## [1.0.1] - 2026-02-17

### 🚀 Features

- Create model folder

### 🐛 Bug Fixes

- Github action update changelog

### 🚜 Refactor

- Addon folder structure
- Tmp fix

### 📚 Documentation

- Update changelog [skip ci]

### ⚙️ Miscellaneous Tasks

- Updates icon and logo

### Fix

- Add permissions to changelog workflow

## [1.0.0] - 2026-02-16

### 🚀 Features

- Add scaffolding and configuration for v1

<!-- generated by git-cliff -->
