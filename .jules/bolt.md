## 2024-05-23 - [Static Data Loading]
**Learning:** File I/O inside hot request paths (like `set_uid`) is a massive bottleneck. Reading and parsing JSON on every request is O(N) at best and creates blocking I/O.
**Action:** Always preload static configuration/data files into memory (dictionary/hash map) at startup for O(1) access.
