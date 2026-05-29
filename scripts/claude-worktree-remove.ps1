param(
    [string]$Root = "C:\ClaudeCodeWorktrees\Pocket_Plus"
)

$ErrorActionPreference = "Stop"

function Write-HookLog([string]$Message) {
    [Console]::Error.WriteLine($Message)
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
$path = [string]$inputJson.worktree_path
$repo = (Resolve-Path -LiteralPath ([string]$inputJson.cwd)).Path

if ([string]::IsNullOrWhiteSpace($path)) {
    exit 0
}

Write-HookLog "Removing Claude Code worktree: $path"
try {
    git -C $repo worktree remove --force $path 2>$null
} catch {
}

Remove-PathIfSafe $path $Root
