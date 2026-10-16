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
- Automatically commit and push completed meaningful changes to GitHub on `development` or a task-specific feature branch (use `codex/` for new feature branches). This is standing user authorization; do not request push approval again unless required by environment permissions.
- Never push directly to `main` or another production/default branch such as `master`. Check the current branch and explicit push destination before pushing. Never merge pull requests without explicit user approval.
- Never force-push, delete remote branches, overwrite remote history, or rewrite shared history. If a push fails, report the error and preserve local commits; do not attempt destructive fixes.
- Run relevant tests before pushing whenever practical. Inspect staged changes to exclude secrets, credentials, and unnecessary generated files.
- If a GitHub remote or authentication is missing, keep completed work committed locally and report what connection setup is needed. Never put tokens in source files, remote URLs, logs, or chat.
- At the end of each task report: summary, files affected, tests/results, branch name, commit hashes, whether all intended changes are committed, and GitHub synchronization status.
- Apply this workflow automatically; the user should not need to request version control again.

## Project scope

Prioritize reliable football person detection/tracking and future normalized field coordinates. Do not add formation classification until detection, tracking, team identification, and coordinate mapping are validated. Do not fabricate formations or confidence scores.

## Code comments — required for existing and future code

- Add and maintain relevant comments and docstrings whenever writing or modifying code. Explain component responsibilities, data formats/units, non-obvious logic, assumptions, tradeoffs, and external API contracts.
- Document public classes/functions where their purpose or behavior needs explanation. In tests, explain unusual fixtures, mock boundaries, and the regression being protected.
- Favor concise explanations of why over comments that merely repeat the code. Keep comments accurate when behavior changes; do not claim unverified accuracy or features.
- Apply this standard to source modules, diagnostic scripts, tests, and configuration. Review nearby comments as part of each future change.

## Verification

Run `.\.venv\Scripts\python.exe -m pytest -q` when relevant. Tests without supplied video/model environment variables should pass without downloading weights or requiring a GPU. The optional integration test uses `FOOTBALL_TEST_VIDEO` and `FOOTBALL_TEST_MODEL`. See README.md for setup and commands.
