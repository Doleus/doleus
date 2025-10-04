# SPDX-FileCopyrightText: 2025 Doleus contributors

# SPDX-License-Identifier: Apache-2.0

# CHANGELOG

## v0.2.2 (2025-10-04)

### Build

- Loosened dependency constraints to support wider range of PyTorch versions (torch>=2.0.0, torchvision>=0.15.0)
  - Fixes compatibility issues with Google Colab and other environments using newer PyTorch versions
  - All tests pass with loosened constraints

## v0.2.1 (2025-10-04) - YANKED

### Note

- **This version was yanked from PyPI** due to untested dependency constraints
- Users should use v0.2.2 or later

## v0.2.0 (2025-10-04)

### Build

- Replaced opencv-python with opencv-python-headless
  ([`5f6b51c`](https://github.com/Doleus/doleus/commit/5f6b51c))

### Features

- Support complex slicing conditions with AND/OR logic
  ([`4e881f5`](https://github.com/Doleus/doleus/commit/4e881f5))

- Added slicing operators: in, not_in, between
  ([`75fa504`](https://github.com/Doleus/doleus/commit/75fa504))

- Added not_between operator
  ([`a4d3442`](https://github.com/Doleus/doleus/commit/a4d3442))

### Documentation

- Added docs for new slicing methods
  ([`50f77d0`](https://github.com/Doleus/doleus/commit/50f77d0))

### Testing

- Added tests for complex slicing conditions (AND/OR operators)
  ([`4e881f5`](https://github.com/Doleus/doleus/commit/4e881f5))

- Added tests for automatically created metadata
  ([`65c10c3`](https://github.com/Doleus/doleus/commit/65c10c3))

- Test for not_between operator
  ([`7cec696`](https://github.com/Doleus/doleus/commit/7cec696))

## v0.1.1 (2025-03-10)

### Bug Fixes

- Bug fix
  ([`ba4c6d6`](https://github.com/iamheinrich/doleus/commit/ba4c6d60b805662ea15bc2a0a6b76197cabc2189))

- Renamed to root_dataset
  ([`bb580b0`](https://github.com/iamheinrich/doleus/commit/bb580b0686b6552bfdc0b8dd493e0e0dcc18bf8e))

- Renaming to root_dataset
  ([`080ec72`](https://github.com/iamheinrich/doleus/commit/080ec72ef20eb8729371587a5610c6c8d4c237de))

### Chores

- Updated gitignore
  ([`475776e`](https://github.com/iamheinrich/doleus/commit/475776e9477a5a503c200d955713cc9c1f505344))

### Documentation

- Object detection demo
  ([`ec52aa7`](https://github.com/iamheinrich/doleus/commit/ec52aa7c8b4a0c5dfef4d568386fb5b4417e816f))

### Refactoring

- Changed docstrings
  ([`45bf991`](https://github.com/iamheinrich/doleus/commit/45bf991b1fa376c9b043a0d486011ea877f3f555))

- Cleaned up codebase
  ([`e9173ce`](https://github.com/iamheinrich/doleus/commit/e9173ced68c0540bc7c2c5fbbe05cd8d44e96baf))

- Complete refactoring
  ([`ce589c5`](https://github.com/iamheinrich/doleus/commit/ce589c51ec982a55db415bc47a85e29b8a591f2d))

- Refactored dataset class
  ([`4d36e9a`](https://github.com/iamheinrich/doleus/commit/4d36e9abb9dcadc9624d47efdd7a6979ecddf207))

- Refactored metric and dataset class
  ([`c2a259d`](https://github.com/iamheinrich/doleus/commit/c2a259d7718d6d2e1bf8aa99e4fd2fe9e3389422))

- Refactored slicing and metadata addition functions
  ([`33a76d8`](https://github.com/iamheinrich/doleus/commit/33a76d8b5e9e5bfe0cec0102de6e43ea9be6e494))

- Remove redundant files
  ([`690cb7c`](https://github.com/iamheinrich/doleus/commit/690cb7ca3bf5987224ae4f3dc9de97e9ec3af42e))

- Removed implicit caching and upload functions
  ([`40f4986`](https://github.com/iamheinrich/doleus/commit/40f498673b0bc2f66f683d4f64afb0362bb0262d))

- Removed redundant files
  ([`c8dccc0`](https://github.com/iamheinrich/doleus/commit/c8dccc0a0584ec7a3b9f734a51da68ab77f7cdd5))

- Removed unnecessairy files
  ([`c27ab5f`](https://github.com/iamheinrich/doleus/commit/c27ab5ff9b8cb852f041d84d96410f2cc2fd4902))

- Restructured metric functions
  ([`8c9dadb`](https://github.com/iamheinrich/doleus/commit/8c9dadbce13ad9604599e022bae484fe08f90cc0))

- Simplified annotation class
  ([`4cbd92a`](https://github.com/iamheinrich/doleus/commit/4cbd92ae0da973150d117926b52a695f41eb7c02))

### Testing

- Adapted tests for dataset class and created conftest file
  ([`e040216`](https://github.com/iamheinrich/doleus/commit/e04021688efea3b85df15d721687b2d008daf775))

- Additional test for dataset
  ([`a1f6a83`](https://github.com/iamheinrich/doleus/commit/a1f6a83888cb4eb2c31fc419512cd6896363df0a))

## v0.1.0 (2025-01-09)

### Bug Fixes

- Corrected handling of predictions
  ([`53d4d03`](https://github.com/iamheinrich/doleus/commit/53d4d032a0ce466072a7e878bc1c673d315b9bda))

### Continuous Integration

- Allow automatic releases
  ([`4ec5a21`](https://github.com/iamheinrich/doleus/commit/4ec5a2182c83856cfef9309a857438041c460b02))

- Bug fix
  ([`7099ecf`](https://github.com/iamheinrich/doleus/commit/7099ecf4616b7ff4399db7eb0c7034909cee9648))

- Remove automatic release
  ([`61dae05`](https://github.com/iamheinrich/doleus/commit/61dae05e5d2d2c663c3d116b7f77acdb46edfa6d))

- Update changelog only
  ([`9b7d939`](https://github.com/iamheinrich/doleus/commit/9b7d939b64455d290ba5d56e6a4fa9b7c6cba37a))

### Features

- Add run method to checks
  ([`6645fcb`](https://github.com/iamheinrich/doleus/commit/6645fcbb005aad6ee2f79b43356d31049847c4f0))

- Added tiny imagenet demo
  ([`7540c7c`](https://github.com/iamheinrich/doleus/commit/7540c7cef5296f86f444ac459af8556793c06ec5))

### Refactoring

- Added code to convert 0-d scalar tensors to 1-d tensors to adhere to annotations.py. dataset.py is
  now only accepting tensors, no lists.
  ([`663b88a`](https://github.com/iamheinrich/doleus/commit/663b88accdb02d2835d050d650aed6b32bff98ba))

- Added more informative error messages
  ([`5ee2b45`](https://github.com/iamheinrich/doleus/commit/5ee2b451d1afc941a57a023a364eee30f4ac03c8))

- Added more validation checks and grouped them in functions
  ([`7c7e0e0`](https://github.com/iamheinrich/doleus/commit/7c7e0e0c6d9e173ce6a5729c849d4800814a3ce9))

- Split moonwatcher into classification and detection object
  ([`d6e786c`](https://github.com/iamheinrich/doleus/commit/d6e786c26a2503f6f7c1a0f0d8c17565a97b184e))

### Testing

- Added test for 0-dimensional scalar tensors
  ([`070f100`](https://github.com/iamheinrich/doleus/commit/070f100171909bfd7bb905bcf6621f3801e8f6ec))

- Updated tests
  ([`0838969`](https://github.com/iamheinrich/doleus/commit/0838969adcb0949cd91c05916771214b29ffc53a))

- Updated tests for annotations.py
  ([`aae01fc`](https://github.com/iamheinrich/doleus/commit/aae01fc611a6252b4a37e390274d90eb5a242999))

## v0.0.0 (2024-10-28)

### Continuous Integration

- Added automatic changelog workflow
  ([`8579ec7`](https://github.com/iamheinrich/doleus/commit/8579ec70826e268a16a435eebc2e426b6060b26e))

### Documentation

- Updated readme to include pre-commit hooks
  ([`24a2237`](https://github.com/iamheinrich/doleus/commit/24a2237c794cd0d6e4ba532e36c99e25d2814626))

### Refactoring

- Consolidate MoonwatcherDataset and MoonwatcherModel into unified Moonwatcher object
  ([`c231db6`](https://github.com/iamheinrich/doleus/commit/c231db6a26d5644b54a0d18b36fc8d2849704cfa))
