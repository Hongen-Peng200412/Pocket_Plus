<#
.SYNOPSIS
    Download remote W&B offline run snapshots and sync them locally.

.DESCRIPTION
    Recursively scans a remote folder for W&B run directories. By default, for
    each wandb/ parent directory it selects only the latest offline-run-* child,
    downloads that run into a local temporary session directory, runs wandb sync
    locally, and then removes the local temporary files.

    The script never deletes remote W&B logs. Syncing an active training run is
    treated as a snapshot sync: only files downloaded at that moment are synced.
#>

param(
    [Parameter(Position = 0)]
    [string]$RemoteRoot,

    [string]$RemoteUser = "penghongen",
    [string]$RemoteHost = "10.102.33.220",
    [int]$RemotePort = 10022,
    [string]$LocalTempDir = $env:TEMP,
    [string]$WandbExe,
    [switch]$AllRuns,
    [switch]$DryRun,
    [switch]$KeepTemp,
    [switch]$IncludeOnlineRuns,
    [string]$Entity,
    [string]$Project,
    [string]$IncludeGlobs,
    [string]$ExcludeGlobs,
    [switch]$MarkSynced
)

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:MSYS2_ARG_CONV_EXCL = "*"

function Write-Step {
    param([string]$Message)
    Write-Host ""
    Write-Host $Message -ForegroundColor Yellow
}

function Resolve-MsysBin {
    $candidates = @(
        $env:PROJECT_MSYS_ROOT,
        $env:MSYS2_ROOT,
        "C:\msys64",
        "D:\msys64"
    ) | Where-Object { -not [string]::IsNullOrWhiteSpace($_) }

    foreach ($root in $candidates) {
        $expandedRoot = [Environment]::ExpandEnvironmentVariables($root.Trim())
        if ($expandedRoot.EndsWith("\usr\bin", [StringComparison]::OrdinalIgnoreCase)) {
            $bin = $expandedRoot
        }
        else {
            $bin = [System.IO.Path]::Combine($expandedRoot, "usr", "bin")
        }
        if (-not [System.IO.Directory]::Exists($bin)) {
            continue
        }

        $required = @("rsync.exe", "cygpath.exe", "sshpass.exe", "ssh.exe")
        $missing = $required | Where-Object { -not [System.IO.File]::Exists([System.IO.Path]::Combine($bin, $_)) }
        if (-not $missing) {
            return $bin
        }
    }
    return $null
}

function Join-RemotePath {
    param([string]$Left, [string]$Right)
    if ($Left.EndsWith("/")) {
        return "$Left$Right"
    }
    return "$Left/$Right"
}

function Quote-Sh {
    param([string]$Value)
    return "'" + ($Value -replace "'", "'\''") + "'"
}

function Get-RemoteParent {
    param([string]$Path)
    $trimmed = $Path.TrimEnd("/")
    $idx = $trimmed.LastIndexOf("/")
    if ($idx -lt 0) {
        return ""
    }
    return $trimmed.Substring(0, $idx)
}

function Get-RemoteLeaf {
    param([string]$Path)
    $trimmed = $Path.TrimEnd("/")
    $idx = $trimmed.LastIndexOf("/")
    if ($idx -lt 0) {
        return $trimmed
    }
    return $trimmed.Substring($idx + 1)
}

function Get-RunTimestampKey {
    param([string]$RunName, [double]$MTime)
    $match = [regex]::Match($RunName, "^(?:offline-)?run-(\d{8})_(\d{6})-")
    if ($match.Success) {
        return "$($match.Groups[1].Value)$($match.Groups[2].Value)"
    }
    return "{0:00000000000000000000}" -f [int64]([Math]::Floor($MTime))
}

function Resolve-WandbExecutable {
    param([string]$ExplicitPath, [int]$SearchTimeoutSeconds = 30)

    $candidates = New-Object System.Collections.Generic.List[string]

    if (-not [string]::IsNullOrWhiteSpace($ExplicitPath)) {
        $candidates.Add($ExplicitPath)
    }

    $pathCmd = Get-Command wandb -ErrorAction SilentlyContinue
    if ($pathCmd -and -not [string]::IsNullOrWhiteSpace($pathCmd.Source)) {
        $candidates.Add($pathCmd.Source)
    }

    $deadline = (Get-Date).AddSeconds($SearchTimeoutSeconds)
    $condaRoots = @(
        $env:CONDA_PREFIX,
        "$env:USERPROFILE\anaconda3",
        "$env:USERPROFILE\miniconda3",
        "D:\Anaconda",
        "D:\anaconda3",
        "D:\miniconda3",
        "C:\ProgramData\anaconda3",
        "C:\ProgramData\miniconda3"
    ) | Where-Object { -not [string]::IsNullOrWhiteSpace($_) } | Select-Object -Unique

    foreach ($root in $condaRoots) {
        if ((Get-Date) -gt $deadline) {
            break
        }
        $expandedRoot = [Environment]::ExpandEnvironmentVariables($root.Trim())
        $candidates.Add((Join-Path $expandedRoot "Scripts\wandb.exe"))
        $envsDir = Join-Path $expandedRoot "envs"
        if ([System.IO.Directory]::Exists($envsDir)) {
            Get-ChildItem -LiteralPath $envsDir -Directory -ErrorAction SilentlyContinue | ForEach-Object {
                if ((Get-Date) -le $deadline) {
                    $candidates.Add((Join-Path $_.FullName "Scripts\wandb.exe"))
                }
            }
        }
    }

    foreach ($candidate in ($candidates | Where-Object { -not [string]::IsNullOrWhiteSpace($_) } | Select-Object -Unique)) {
        if (Test-Path -LiteralPath $candidate) {
            return (Resolve-Path -LiteralPath $candidate).Path
        }
    }
    return $null
}

function Invoke-RemoteCommand {
    param([string]$Command)
    & $script:SshPassExe -f $script:SshPassFilePosix $script:SshExe `
        -p $RemotePort `
        -o StrictHostKeyChecking=accept-new `
        -o WarnWeakCrypto=no `
        -o PreferredAuthentications=password `
        -o PubkeyAuthentication=no `
        -o NumberOfPasswordPrompts=1 `
        "$RemoteUser@$RemoteHost" `
        "bash -lc $(Quote-Sh $Command)"
}

function Convert-ToPosixPath {
    param([string]$WindowsPath)
    return (& $script:CygpathExe -u $WindowsPath).Trim()
}

function Get-RemoteRuns {
    if ($IncludeOnlineRuns) {
        $nameExpr = "\( -name 'offline-run-*' -o \( -name 'run-*' -a -path '*/wandb/run-*' \) \)"
    }
    else {
        $nameExpr = "-name 'offline-run-*'"
    }
    $findCmd = "find $(Quote-Sh $RemoteRoot) -type d $nameExpr -printf '%T@`t%p\n' 2>/dev/null"
    $raw = Invoke-RemoteCommand $findCmd
    if ($LASTEXITCODE -ne 0) {
        throw "Remote find failed under $RemoteRoot"
    }

    $runs = @()
    foreach ($line in ($raw -split "`n")) {
        $trimmed = $line.Trim()
        if ([string]::IsNullOrWhiteSpace($trimmed)) {
            continue
        }
        $parts = $trimmed -split "`t", 2
        if ($parts.Count -ne 2) {
            continue
        }
        $mtime = 0.0
        [void][double]::TryParse($parts[0], [System.Globalization.NumberStyles]::Float, [System.Globalization.CultureInfo]::InvariantCulture, [ref]$mtime)
        $remotePath = $parts[1]
        $runName = Get-RemoteLeaf $remotePath
        $parent = Get-RemoteParent $remotePath
        $runs += [pscustomobject]@{
            RemotePath = $remotePath
            Parent     = $parent
            Name       = $runName
            MTime      = $mtime
            SortKey    = Get-RunTimestampKey -RunName $runName -MTime $mtime
        }
    }
    return $runs
}

function Select-RunsToSync {
    param([object[]]$Runs)
    if ($AllRuns) {
        return $Runs | Sort-Object Parent, SortKey, MTime
    }

    $selected = @()
    foreach ($group in ($Runs | Group-Object Parent)) {
        $latest = $group.Group | Sort-Object SortKey, MTime -Descending | Select-Object -First 1
        $selected += $latest
    }
    return $selected | Sort-Object Parent
}

function Copy-RemoteRun {
    param([object]$Run, [int]$Index, [string]$SessionTempDir)

    $localParent = Join-Path $SessionTempDir ("run_{0:0000}" -f $Index)
    $localRun = Join-Path $localParent $Run.Name
    New-Item -ItemType Directory -Path $localRun -Force | Out-Null

    $localRunPosix = Convert-ToPosixPath $localRun
    $remoteRunWithSlash = Join-RemotePath $Run.RemotePath ""
    $remoteSource = "${RemoteUser}@${RemoteHost}:$(Quote-Sh $remoteRunWithSlash)"
    Write-Host "  Downloading $($Run.Name)"
    Write-Host "    from: $($Run.RemotePath)" -ForegroundColor DarkGray

    & $script:RsyncExe -a --partial `
        -e $script:RemoteShell `
        $remoteSource `
        "$localRunPosix/"

    $exitCode = $LASTEXITCODE
    if ($exitCode -ne 0 -and $exitCode -ne 24) {
        throw "rsync failed for $($Run.RemotePath) with exit code $exitCode"
    }
    if ($exitCode -eq 24) {
        Write-Host "    Warning: rsync saw vanished files; treating active-run snapshot as usable." -ForegroundColor Yellow
    }
    return $localRun
}

function Invoke-WandbSyncCommand {
    param([string]$Executable, [string[]]$Arguments)

    $previousErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        $output = & $Executable @Arguments 2>&1
        $exitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $previousErrorActionPreference
    }

    return [pscustomobject]@{
        ExitCode = $exitCode
        Output   = @($output)
    }
}

function Test-WandbMetadataAccessDenied {
    param([object[]]$Output)
    $text = ($Output | ForEach-Object { [string]$_ }) -join "`n"
    return (
        $text -match "wandb-metadata\.json" -and
        ($text -match "HTTP 403" -or $text -match "AccessDenied" -or $text -match "storage\.objects\.delete")
    )
}

function Add-WandbMetadataExclude {
    param([string[]]$Arguments)
    $updated = New-Object System.Collections.Generic.List[string]
    $excludeSeen = $false

    for ($i = 0; $i -lt $Arguments.Count; $i++) {
        $arg = $Arguments[$i]
        $updated.Add($arg)
        if ($arg -eq "--exclude-globs" -and ($i + 1) -lt $Arguments.Count) {
            $excludeSeen = $true
            $i += 1
            $existing = $Arguments[$i]
            if ($existing -notmatch "(^|,)wandb-metadata\.json(,|$)") {
                $existing = "$existing,wandb-metadata.json"
            }
            $updated.Add($existing)
        }
    }

    if (-not $excludeSeen) {
        $runPath = $updated[$updated.Count - 1]
        $updated.RemoveAt($updated.Count - 1)
        $updated.Add("--exclude-globs")
        $updated.Add("wandb-metadata.json")
        $updated.Add($runPath)
    }

    return $updated.ToArray()
}

if ([string]::IsNullOrWhiteSpace($RemoteRoot)) {
    Write-Host "Usage:" -ForegroundColor Cyan
    Write-Host "  & .\sync_wandb_remote.bat /home/penghongen/My_Project/feedback_plus/logs/CPC/CPC_main____job297517"
    Write-Host "  .\sync_wandb_remote.ps1 -RemoteRoot /home/... [-DryRun] [-AllRuns] [-IncludeOnlineRuns] [-Entity ENTITY] [-Project PROJECT]"
    exit 2
}

$RemoteRoot = $RemoteRoot.Trim().TrimEnd("/")
if (-not $RemoteRoot.StartsWith("/", [StringComparison]::Ordinal)) {
    if ($RemoteRoot -match "^(home|tmp|mnt|data|scratch)/") {
        $RemoteRoot = "/$RemoteRoot"
        Write-Host "Warning: RemoteRoot did not start with '/'; normalized to $RemoteRoot" -ForegroundColor Yellow
    }
    else {
        Write-Host "Error: RemoteRoot must be an absolute Linux path, for example /home/penghongen/..." -ForegroundColor Red
        Write-Host "Got: $RemoteRoot" -ForegroundColor Red
        exit 2
    }
}
$MsysBin = Resolve-MsysBin
if (-not $MsysBin) {
    Write-Host "Error: MSYS2 toolchain not found." -ForegroundColor Red
    Write-Host "Set PROJECT_MSYS_ROOT or MSYS2_ROOT, or install MSYS2 at C:\msys64 or D:\msys64." -ForegroundColor Red
    exit 1
}

$script:RsyncExe = Join-Path $MsysBin "rsync.exe"
$script:CygpathExe = Join-Path $MsysBin "cygpath.exe"
$script:SshPassExe = Join-Path $MsysBin "sshpass.exe"
$script:SshExe = Join-Path $MsysBin "ssh.exe"
$SshPassFile = Join-Path $env:USERPROFILE ".ssh\pocket_plus_sshpass.txt"

if (-not (Test-Path -LiteralPath $SshPassFile)) {
    Write-Host "Error: SSH password file not found: $SshPassFile" -ForegroundColor Red
    exit 1
}

$env:Path = "$MsysBin;$env:Path"
$script:SshPassFilePosix = Convert-ToPosixPath $SshPassFile
$script:RemoteShell = "sshpass -f $script:SshPassFilePosix ssh -p $RemotePort -o StrictHostKeyChecking=accept-new -o WarnWeakCrypto=no -o PreferredAuthentications=password -o PubkeyAuthentication=no -o NumberOfPasswordPrompts=1"

$ResolvedWandbExe = Resolve-WandbExecutable -ExplicitPath $WandbExe -SearchTimeoutSeconds 30
if (-not $ResolvedWandbExe -and -not $DryRun) {
    Write-Host "Error: wandb executable not found within the bounded search." -ForegroundColor Red
    Write-Host "Pass -WandbExe explicitly, or install/login W&B in a local Python/Conda environment." -ForegroundColor Red
    exit 1
}

$sessionName = "pocket_plus_wandb_sync_{0:yyyyMMdd_HHmmss}" -f (Get-Date)
$SessionTempDir = Join-Path $LocalTempDir $sessionName

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host " Remote W&B snapshot sync" -ForegroundColor Cyan
Write-Host " Remote: ${RemoteUser}@${RemoteHost}:${RemotePort}" -ForegroundColor Cyan
Write-Host " Remote root: $RemoteRoot" -ForegroundColor Cyan
Write-Host " Mode: $(if ($DryRun) { 'dry-run' } else { 'download + local wandb sync' })" -ForegroundColor Cyan
Write-Host " Selection: $(if ($AllRuns) { 'all matched runs' } else { 'latest run per wandb parent' })" -ForegroundColor Cyan
if (-not [string]::IsNullOrWhiteSpace($Entity)) {
    Write-Host " Entity: $Entity" -ForegroundColor Cyan
}
if (-not [string]::IsNullOrWhiteSpace($Project)) {
    Write-Host " Project: $Project" -ForegroundColor Cyan
}
if ($ResolvedWandbExe) {
    Write-Host " WandB: $ResolvedWandbExe" -ForegroundColor Cyan
}
Write-Host "==========================================================" -ForegroundColor Cyan

try {
    Write-Step "[1/4] Finding remote W&B run directories..."
    $runs = @(Get-RemoteRuns)
    if ($runs.Count -eq 0) {
        Write-Host "No matching W&B run directories found." -ForegroundColor Green
        exit 0
    }
    $selectedRuns = @(Select-RunsToSync -Runs $runs)
    Write-Host "Found $($runs.Count) matching run(s); selected $($selectedRuns.Count)." -ForegroundColor Green
    foreach ($run in $selectedRuns) {
        Write-Host "  $($run.RemotePath)"
    }

    if ($DryRun) {
        Write-Host ""
        Write-Host "Dry run complete. No files were downloaded or synced." -ForegroundColor Green
        exit 0
    }

    Write-Step "[2/4] Creating local temporary session..."
    New-Item -ItemType Directory -Path $SessionTempDir -Force | Out-Null
    Set-Content -LiteralPath (Join-Path $SessionTempDir ".created_by_sync_wandb_remote") -Value "temporary wandb sync session" -Encoding ASCII
    Write-Host "Temp: $SessionTempDir" -ForegroundColor Green

    Write-Step "[3/4] Downloading selected run snapshots..."
    $downloaded = @()
    $index = 0
    foreach ($run in $selectedRuns) {
        $index += 1
        $localPath = Copy-RemoteRun -Run $run -Index $index -SessionTempDir $SessionTempDir
        $downloaded += [pscustomobject]@{
            LocalPath  = $localPath
            RemotePath = $run.RemotePath
            Name       = $run.Name
        }
    }

    Write-Step "[4/4] Running local wandb sync..."
    $synced = 0
    $failed = 0
    foreach ($run in $downloaded) {
        Write-Host "  Syncing $($run.Name)"
        $syncArgs = @("sync", "--no-include-online")
        if ($MarkSynced) {
            $syncArgs += "--mark-synced"
        }
        else {
            $syncArgs += "--no-mark-synced"
        }
        if (-not [string]::IsNullOrWhiteSpace($Entity)) {
            $syncArgs += @("--entity", $Entity)
        }
        if (-not [string]::IsNullOrWhiteSpace($Project)) {
            $syncArgs += @("--project", $Project)
        }
        if (-not [string]::IsNullOrWhiteSpace($IncludeGlobs)) {
            $syncArgs += @("--include-globs", $IncludeGlobs)
        }
        if (-not [string]::IsNullOrWhiteSpace($ExcludeGlobs)) {
            $syncArgs += @("--exclude-globs", $ExcludeGlobs)
        }
        $syncArgs += $run.LocalPath

        $syncResult = Invoke-WandbSyncCommand -Executable $ResolvedWandbExe -Arguments $syncArgs
        $syncExitCode = $syncResult.ExitCode
        $syncOutput = $syncResult.Output

        if ($syncExitCode -ne 0 -and [string]::IsNullOrWhiteSpace($ExcludeGlobs) -and (Test-WandbMetadataAccessDenied -Output $syncOutput)) {
            Write-Host "    W&B returned metadata 403 after a partial sync; retrying once without wandb-metadata.json." -ForegroundColor Yellow
            $retryArgs = Add-WandbMetadataExclude -Arguments $syncArgs
            $syncResult = Invoke-WandbSyncCommand -Executable $ResolvedWandbExe -Arguments $retryArgs
            $syncExitCode = $syncResult.ExitCode
            $syncOutput = $syncResult.Output
            $syncArgs = $retryArgs
        }

        if ($syncExitCode -eq 0) {
            $synced += 1
            Write-Host "    OK" -ForegroundColor Green
            $syncOutput | ForEach-Object {
                $match = [regex]::Match([string]$_, "(https://wandb\.ai/[^\s]+)")
                if ($match.Success) {
                    Write-Host "    $($match.Value)" -ForegroundColor Cyan
                }
            }
        }
        else {
            $failed += 1
            $script:SyncHadFailure = $true
            Write-Host "    FAILED" -ForegroundColor Red
            Write-Host "    Command: $ResolvedWandbExe $($syncArgs -join ' ')" -ForegroundColor DarkGray
            $syncOutput | ForEach-Object { Write-Host "      $_" -ForegroundColor Red }
        }
    }

    Write-Host ""
    Write-Host "Sync finished: $synced succeeded, $failed failed." -ForegroundColor $(if ($failed -eq 0) { "Green" } else { "Yellow" })
    if ($failed -gt 0) {
        exit 1
    }
}
finally {
    if (-not $DryRun -and -not $KeepTemp -and -not $script:SyncHadFailure -and (Test-Path -LiteralPath $SessionTempDir)) {
        $sentinel = Join-Path $SessionTempDir ".created_by_sync_wandb_remote"
        if (Test-Path -LiteralPath $sentinel) {
            Write-Host ""
            Write-Host "Cleaning local temp: $SessionTempDir" -ForegroundColor Gray
            Remove-Item -LiteralPath $SessionTempDir -Recurse -Force -ErrorAction SilentlyContinue
        }
        else {
            Write-Host "Temp cleanup skipped because sentinel file is missing: $SessionTempDir" -ForegroundColor Yellow
        }
    }
    elseif (-not $DryRun -and $KeepTemp) {
        Write-Host "Keeping local temp: $SessionTempDir" -ForegroundColor Yellow
    }
    elseif (-not $DryRun -and $script:SyncHadFailure -and (Test-Path -LiteralPath $SessionTempDir)) {
        Write-Host "Keeping local temp after sync failure: $SessionTempDir" -ForegroundColor Yellow
    }
}
