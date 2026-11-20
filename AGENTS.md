# Starting a task

1. Read the user's task and any linked assignment brief. If it references an
   existing issue in this repository, read the issue and check its open blockers.
   A task given directly in chat is sufficient to start.
2. Run `git status --short --branch` and `git worktree list` before editing.
   Preserve existing changes and reuse the worktree already assigned to the task.
3. For a new task that edits files, use a dedicated branch and worktree. When
   starting from a clean `main` worktree, refresh it with
   `git pull --ff-only origin main` before branching. If the remote is
   unavailable, use the local base and report that it could not be refreshed.
   Name branches `<type>/<short-description>`, or
   `<type>/<issue-number>/<short-description>` when an issue exists.
4. Inspect the relevant files and available checks before making changes.
   Take scope and acceptance criteria from the current assignment.

# Checking and publishing changes

- Run the relevant checks and review the diff before handing off. For changes to
  `agent.py`, check Python syntax and use mocked responses for agent-loop checks
  when a live model endpoint is unavailable.
- Keep credentials, private endpoints and internal project details out of
  tracked files and anything published to GitHub. Use placeholder values in
  examples.
- Create public issues, push branches and open pull requests when requested.
  Use a pull request for publication unless the user requests a direct push.
