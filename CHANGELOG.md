# CHANGELOG


## v0.0.0 (2024-10-28)

### Continuous Integration

* ci: added automatic changelog workflow ([`8579ec7`](https://github.com/iamheinrich/doleus/commit/8579ec70826e268a16a435eebc2e426b6060b26e))

### Documentation

* docs: updated readme to include pre-commit hooks ([`24a2237`](https://github.com/iamheinrich/doleus/commit/24a2237c794cd0d6e4ba532e36c99e25d2814626))

### Refactoring

* refactor: consolidate MoonwatcherDataset and MoonwatcherModel into unified Moonwatcher object ([`c231db6`](https://github.com/iamheinrich/doleus/commit/c231db6a26d5644b54a0d18b36fc8d2849704cfa))

### Unknown

* Merge pull request #14 from iamheinrich/readme

docs: updated readme to include pre-commit hooks ([`e1c76e5`](https://github.com/iamheinrich/doleus/commit/e1c76e50a048c64849ec78e523be9c5c3ed4c123))

* Merge pull request #12 from iamheinrich/1_create_model_object

Restructure model object ([`6577abd`](https://github.com/iamheinrich/doleus/commit/6577abd6f949b3ea9d56bccdcf69c596fbf61b28))

* Resolve merge conflicts ([`def078e`](https://github.com/iamheinrich/doleus/commit/def078e88eecf2000d872306e0111130ad45fe87))

* Adapted the Model class so that it automatically converts predictions into the Predictions object. Object detection is still missing and no tests are written yet. ([`52ee283`](https://github.com/iamheinrich/doleus/commit/52ee283527b9fd9058b27fa7357171e5a8694fc4))

* Managed to get one test to work. Simply run the breast cancer demo. However, there are still quite a few todos. We need to bring the predictions that the user passes to the dataset into the required format. That will likely be the biggeset task, because we want to do it in a smart, clean and understandable way. Moreover, there's still quite a bit of clean up to do. Committing for review. ([`feaf95c`](https://github.com/iamheinrich/doleus/commit/feaf95c568e7434e5f28b32049477dd2b3b464ee))

* Created breast cancer detection demo. Labels class was changed to accept more than just a 1-d tensor to allow for multi-class classification.-0 ([`542838f`](https://github.com/iamheinrich/doleus/commit/542838f4e8e04a0eeed78d087d12167564ad820b))

* Merge pull request #1 from moonwatcher-ai/feature/add-class-metrics

Feature/add class metrics ([`fe8af69`](https://github.com/iamheinrich/doleus/commit/fe8af69cabecfdbbbf4277e073ae0f84a6bb7259))

* Added groundtruth-based slicing ([`da769cb`](https://github.com/iamheinrich/doleus/commit/da769cbc574db08af04ae827bf618e1f6fc0d871))

* Added per-class metrics for classification metrics ([`a9b6705`](https://github.com/iamheinrich/doleus/commit/a9b670596f064e5a171d3e5c325c529583dfb85f))

* Added per-class metrics for mAP ([`873a885`](https://github.com/iamheinrich/doleus/commit/873a8850209b3ebc176efe3f43a4aeca9630178c))

* Added per-class metrics ([`1e8c433`](https://github.com/iamheinrich/doleus/commit/1e8c43359297beefa9c4c0cdec5eda292655bcf8))

* 0.1.0-alpha ([`13f8dc4`](https://github.com/iamheinrich/doleus/commit/13f8dc4a73df92cb9e1da34092f46e597f5affc4))
