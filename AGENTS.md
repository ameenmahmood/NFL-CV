# Project instructions

## Git workflow — required for every development task

- Use the existing Git repository. Initialize one only if none exists.
- Before modifying files, run `git status` and inspect existing changes. Never overwrite, discard, reset, or include unrelated user changes without explicit permission. Preserve incomplete work and report its uncommitted status.
- Track relevant source code, configuration, documentation, and project assets. Keep videos, model weights, virtual environments, generated outputs, and secrets out of Git.
- Commit after each meaningful completed feature, fix, refactoring, test addition, or documentation change. Split complex work into logical commits, and commit completed work before starting a separate task. Do not wait until the end of a long session.
- Use descriptive conventional commit messages: `feat:`, `fix:`, `refactor:`, `docs:`, `test:`, or `chore:`. Explain the concrete change and purpose.
- Before committing, review `git status`, `git diff`, and the staged diff. Run relevant tests whenever practical, check the staged files for secrets and unnecessary generated files, and stage only files belonging to the current change. Do not modify unrelated user changes.
- Maintain root `CHANGELOG.md` for every meaningful completed change: date, feature/fix/modification, summary, relevant files, and associated Git commit hash. Use the actual date of the change; identify retrospective baseline entries when appropriate.
- If a hash is unavailable while drafting an entry, update the changelog after creating the implementation commit, then commit the changelog update. Do not amend or rewrite history to embed a commit's own hash; the follow-up changelog synchronization commit can be found with `git log -- CHANGELOG.md` and does not require a recursive self-referencing entry.
- Keep intended completed changes committed locally. Never push to GitHub without explicit user authorization. Never force-push, rewrite shared history, or delete branches without permission.
- At the end of each task report: summary, files affected, tests/results, commit hashes, whether all intended changes are committed, and whether anything was pushed.
- Apply this workflow automatically; the user should not need to request version control again.

## Project scope

Prioritize reliable football person detection/tracking and future normalized field coordinates. Do not add formation classification until detection, tracking, team identification, and coordinate mapping are validated. Do not fabricate formations or confidence scores.

## Verification

Run `.\.venv\Scripts\python.exe -m pytest -q` when relevant. Tests without supplied video/model environment variables should pass without downloading weights or requiring a GPU. The optional integration test uses `FOOTBALL_TEST_VIDEO` and `FOOTBALL_TEST_MODEL`. See README.md for setup and commands.
