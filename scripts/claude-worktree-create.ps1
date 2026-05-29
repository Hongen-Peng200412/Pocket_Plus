param(
    [string]$Root = "C:\ClaudeCodeWorktrees\Pocket_Plus"
)

$ErrorActionPreference = "Stop"

function Write-HookLog([string]$Message) {
    [Console]::Error.WriteLine($Message)
}

function Convert-ToSafeName([string]$Name) {
    $safe = $Name -replace '[^A-Za-z0-9_.-]', '-'
    if ([string]::IsNullOrWhiteSpace($safe)) {
        $safe = "agent"
    }
    return $safe
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

function Remove-PathIfSafe([string]$Path, [string]$AllowedRoot) {
    if (-not (Test-Path -LiteralPath $Path)) {
        return
    }
    $fullPath = [System.IO.Path]::GetFullPath($Path).TrimEnd("\")
    $fullRoot = [System.IO.Path]::GetFullPath($AllowedRoot).TrimEnd("\")
    if (-not (($fullPath + "\").StartsWith($fullRoot + "\", [System.StringComparison]::OrdinalIgnoreCase))) {
        throw "Refusing to delete outside worktree root: $fullPath"
    }
    Clear-ReadonlyAttributes $fullPath
    Remove-Item -LiteralPath $fullPath -Recurse -Force
}

$stdin = [Console]::In.ReadToEnd()
$inputJson = $stdin | ConvertFrom-Json
$name = Convert-ToSafeName ([string]$inputJson.name)
$repo = (Resolve-Path -LiteralPath ([string]$inputJson.cwd)).Path

New-Item -ItemType Directory -Force -Path $Root | Out-Null

$target = Join-Path $Root $name
if (Test-Path -LiteralPath $target) {
    Write-HookLog "Claude worktree target already exists; removing stale target: $target"
    try {
        git -C $repo worktree remove --force $target 2>$null
    } catch {
    }
    Remove-PathIfSafe $target $Root
}

Write-HookLog "Creating Claude Code worktree outside OneDrive: $target"
git -C $repo worktree add --detach $target HEAD 1>$null
if ($LASTEXITCODE -ne 0) {
    throw "git worktree add failed for $target"
}

[Console]::Out.WriteLine((Resolve-Path -LiteralPath $target).Path)
