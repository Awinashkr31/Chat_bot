# Contributing

Thank you for contributing to the MCA Chatbot project! 

## Data Privacy & Handling
**CRITICAL**: Real student data must **never** be committed to this repository. This repository is public and contains no real student Personally Identifiable Information (PII).

When running the project locally:
1. Do not use the real `students.json` file inside the repository tree unless you have added it to `.gitignore` and are certain it won't be committed.
2. The recommended approach is to use `students.example.json` as a template for your local testing or point the `STUDENTS_DATA_PATH` environment variable to a file outside the repository.
3. If you accidentally commit real PII, you must use `git filter-repo` to purge it from the history. Reverting the commit is not sufficient.

## Development Setup
1. Create a virtual environment and activate it.
2. Install dependencies: `pip install -r requirements.txt`
3. Run tests using `pytest` to ensure your changes didn't break existing functionality.
