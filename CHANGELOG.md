# CHANGELOG


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
