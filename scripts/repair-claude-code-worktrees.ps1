param(
    [string]$Repo = (Get-Location).Path,
    [switch]$PreflightForClaudeAgents
)

$ErrorActionPreference = "Stop"

function Get-FullPath([string]$Path) {
    return [System.IO.Path]::GetFullPath($Path).TrimEnd("\")
}

function Test-UnderPath([string]$Child, [string]$Parent) {
    $childFull = Get-FullPath $Child
    $parentFull = Get-FullPath $Parent
    return ($childFull + "\").StartsWith($parentFull + "\", [System.StringComparison]::OrdinalIgnoreCase)
}

function Clear-ReadonlyAttributes([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path)) {
        return
    }
    Get-ChildItem -LiteralPath $Path -Force -Recurse -ErrorAction SilentlyContinue | ForEach-Object {
        try {
            $_.Attributes = $_.Attributes -band (-bnot [System.IO.FileAttributes]::ReadOnly)
        } catch {
        }
    }
    try {
        $item = Get-Item -LiteralPath $Path -Force
        $item.Attributes = $item.Attributes -band (-bnot [System.IO.FileAttributes]::ReadOnly)
    } catch {
    }
}

function Get-WorktreeStatus([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path)) {
        return @()
    }
    $status = git -C $Path status --porcelain 2>$null
    if ($LASTEXITCODE -ne 0) {
        return @("__GIT_STATUS_FAILED__")
    }
    return @($status)
}

function Test-OnlyClaudeLocalChanges([string[]]$StatusLines) {
    foreach ($line in $StatusLines) {
        if (-not $line) {
            continue
        }
        if ($line -eq "__GIT_STATUS_FAILED__") {
            return $false
        }
        if ($line -notmatch '^\s*[MADRCU?!]{1,2}\s+\.claude[\\/](settings\.local\.json|settings\.json)$') {
            return $false
        }
    }
    return $true
}

function Remove-DirectorySafe([string]$Path, [string]$AllowedRoot) {
    if (-not (Test-Path -LiteralPath $Path)) {
        return
    }
    $resolvedPath = (Resolve-Path -LiteralPath $Path).Path
    $resolvedRoot = (Resolve-Path -LiteralPath $AllowedRoot).Path
    if (-not (Test-UnderPath $resolvedPath $resolvedRoot)) {
        throw "Refusing to delete outside allowed root: $resolvedPath"
    }
    Clear-ReadonlyAttributes $resolvedPath
    Remove-Item -LiteralPath $resolvedPath -Recurse -Force
}

function Get-GitWorktrees {
    $raw = git -C $Repo worktree list --porcelain
    $items = @()
    $current = $null
    foreach ($line in $raw) {
        if ($line -match '^worktree (.+)$') {
            if ($null -ne $current) {
                $items += [pscustomobject]$current
            }
            $current = @{
                Path = $Matches[1].Replace("/", "\")
                Locked = $false
                LockReason = ""
            }
        } elseif ($null -ne $current -and $line -match '^locked ?(.*)$') {
            $current.Locked = $true
            $current.LockReason = $Matches[1]
        }
    }
    if ($null -ne $current) {
        $items += [pscustomobject]$current
    }
    return @($items)
}

$Repo = (Resolve-Path -LiteralPath $Repo).Path
$ClaudeWorktreeRoot = Join-Path $Repo ".claude\worktrees"
$GitCommonDir = Get-FullPath (git -C $Repo rev-parse --path-format=absolute --git-common-dir)
$GitWorktreesDir = Join-Path $GitCommonDir "worktrees"

Write-Host "Repo: $Repo"

$worktrees = Get-GitWorktrees
foreach ($worktree in $worktrees) {
    $path = Get-FullPath $worktree.Path
    if (-not (Test-UnderPath $path $ClaudeWorktreeRoot)) {
        continue
    }

    $pidAlive = $false
    if ($worktree.LockReason -match 'pid\s+(\d+)') {
        $pidAlive = [bool](Get-Process -Id ([int]$Matches[1]) -ErrorAction SilentlyContinue)
    }

    $status = Get-WorktreeStatus $path
    $safeToRemove = Test-OnlyClaudeLocalChanges $status
    $staleLock = $worktree.Locked -and (-not $pidAlive)
    $missingPath = -not (Test-Path -LiteralPath $path)
    $preflightClean = $PreflightForClaudeAgents -and $safeToRemove -and (-not $pidAlive)

    if ($missingPath -or $staleLock -or $preflightClean) {
        if (-not $safeToRemove -and -not $missingPath) {
            Write-Host "SKIP dirty Claude worktree: $path"
            $status | ForEach-Object { Write-Host "  $_" }
            continue
        }
        Write-Host "Remove Claude worktree: $path"
        try {
            git -C $Repo worktree unlock $path 2>$null
        } catch {
        }
        try {
            git -C $Repo worktree remove --force $path 2>$null
        } catch {
        }
        if (Test-Path -LiteralPath $path) {
            Remove-DirectorySafe $path $ClaudeWorktreeRoot
        }
    } else {
        Write-Host "Keep active/registered Claude worktree: $path"
    }
}

if (Test-Path -LiteralPath $ClaudeWorktreeRoot) {
    $registered = @(Get-GitWorktrees | ForEach-Object { Get-FullPath $_.Path })
    foreach ($dir in Get-ChildItem -LiteralPath $ClaudeWorktreeRoot -Force -Directory) {
        $dirPath = Get-FullPath $dir.FullName
        if ($registered -contains $dirPath) {
            continue
        }
        $status = Get-WorktreeStatus $dirPath
        if (Test-OnlyClaudeLocalChanges $status) {
            Write-Host "Remove orphan Claude worktree dir: $dirPath"
            Remove-DirectorySafe $dirPath $ClaudeWorktreeRoot
        } else {
            Write-Host "SKIP dirty orphan Claude dir: $dirPath"
            $status | ForEach-Object { Write-Host "  $_" }
        }
    }
}

git -C $Repo worktree prune

if (Test-Path -LiteralPath $GitWorktreesDir) {
    foreach ($meta in Get-ChildItem -LiteralPath $GitWorktreesDir -Force -Directory) {
        $gitdirFile = Join-Path $meta.FullName "gitdir"
        $removeMeta = $false

        if (-not (Test-Path -LiteralPath $gitdirFile)) {
            $removeMeta = $true
        } else {
            $gitdir = (Get-Content -LiteralPath $gitdirFile -First 1).Trim().Replace("/", "\")
            $worktreePath = Split-Path $gitdir -Parent
            if ((Test-UnderPath $worktreePath $ClaudeWorktreeRoot) -and (-not (Test-Path -LiteralPath $worktreePath))) {
                $removeMeta = $true
            }
        }

        if ($removeMeta) {
            Write-Host "Remove stale Git worktree metadata: $($meta.FullName)"
            Remove-DirectorySafe $meta.FullName $GitWorktreesDir
        }
    }

    if (-not (Get-ChildItem -LiteralPath $GitWorktreesDir -Force -ErrorAction SilentlyContinue)) {
        Remove-Item -LiteralPath $GitWorktreesDir -Force -ErrorAction SilentlyContinue
    }
}

git -C $Repo worktree prune

if ($PreflightForClaudeAgents -and (Test-Path -LiteralPath $ClaudeWorktreeRoot)) {
    if (-not (Get-ChildItem -LiteralPath $ClaudeWorktreeRoot -Force -ErrorAction SilentlyContinue)) {
        Remove-Item -LiteralPath $ClaudeWorktreeRoot -Force
        Write-Host "Removed empty .claude/worktrees parent to avoid Claude Code EEXIST."
    } else {
        Write-Host "WARNING: .claude/worktrees is not empty; Claude Code EnterWorktree may still hit EEXIST."
    }
}

Write-Host ""
Write-Host "Final git worktrees:"
git -C $Repo worktree list
Write-Host "OK"
