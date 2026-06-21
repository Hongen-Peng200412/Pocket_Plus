# Claude Code agent/worktree preflight

Before launching Claude Code subagents or `EnterWorktree` in this project, run:

```powershell
& ".\scripts\repair-claude-code-worktrees.ps1" -PreflightForClaudeAgents
```

If this command ends with `OK`, proceed.

If Claude Code still reports `EEXIST: file already exists, mkdir '.claude\worktrees'`, do not retry the Agent tool repeatedly. Treat it as a Claude Code worktree-tool bug and immediately switch to one of these modes:

1. Main-session implementation with explicit non-overlapping file scopes.
2. Manual `git worktree add` directories, followed by direct shell work in those directories, not the Agent tool.

Known failure pattern:

- `.claude/worktrees` exists as a normal parent directory.
- Claude Code `EnterWorktree` tries to create that parent directory again instead of accepting it.
- Agent calls may still invoke the same internal worktree path even when given an existing manual worktree.

Do not spend tokens repeatedly inspecting `.claude/worktrees` after the preflight command has already reported `OK`.
