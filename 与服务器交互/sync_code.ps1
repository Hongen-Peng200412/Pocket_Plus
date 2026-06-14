<#
.SYNOPSIS
    Safe sync Pocket_Plus to the remote server.

.DESCRIPTION
    1. Keep the remote Pocket_Plus directory to avoid disturbing running jobs.
    2. Verify rsync exists on the remote server.
    3. Remove any existing remote .git directory.
    4. Upload local files with rsync while excluding local-only files.
    5. Reset permissions and fix .sbatch / .sh line endings.
#>

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$LocalPath = (Resolve-Path (Join-Path $ScriptDir "..")).Path
$RemoteUser = "penghongen"
$RemoteIP = "10.102.33.220"
$RemotePort = "10022"
$RemoteBaseDir = "/home/penghongen/My_Project"
$RemoteTargetDir = "Pocket_Plus"

function Resolve-MsysBin {
    $Candidates = @(
        $env:PROJECT_MSYS_ROOT,
        $env:MSYS2_ROOT,
        "C:\msys64",
        "D:\msys64"
    ) | Where-Object { -not [string]::IsNullOrWhiteSpace($_) }

    foreach ($Root in $Candidates) {
        $ExpandedRoot = [Environment]::ExpandEnvironmentVariables($Root.Trim())
        if ($ExpandedRoot.EndsWith("\usr\bin", [StringComparison]::OrdinalIgnoreCase)) {
            $Bin = $ExpandedRoot
        }
        else {
            $Bin = [System.IO.Path]::Combine($ExpandedRoot, "usr", "bin")
        }
        if (-not [System.IO.Directory]::Exists($Bin)) {
            continue
        }

        $Required = @("rsync.exe", "cygpath.exe", "sshpass.exe", "ssh.exe")
        $Missing = $Required | Where-Object { -not [System.IO.File]::Exists([System.IO.Path]::Combine($Bin, $_)) }
        if (-not $Missing) {
            return $Bin
        }
    }

    return $null
}

$MsysBin = Resolve-MsysBin
if (-not $MsysBin) {
    Write-Host "Error: MSYS2 toolchain not found." -ForegroundColor Red
    Write-Host "Set PROJECT_MSYS_ROOT or MSYS2_ROOT, or install MSYS2 at C:\msys64 or D:\msys64." -ForegroundColor Red
    exit 1
}

$RsyncExe = Join-Path $MsysBin "rsync.exe"
$CygpathExe = Join-Path $MsysBin "cygpath.exe"
$SshPassExe = Join-Path $MsysBin "sshpass.exe"
$SshExe = Join-Path $MsysBin "ssh.exe"
$SshPassFile = Join-Path $env:USERPROFILE ".ssh\pocket_plus_sshpass.txt"
$RemoteFullDir = "$RemoteBaseDir/$RemoteTargetDir"
$RemoteSpec = "${RemoteUser}@${RemoteIP}:${RemoteFullDir}/"

function Invoke-RemoteCommand {
    param([string]$Command)
    & $SshPassExe -f $SshPassFilePosix $SshExe -p $RemotePort -o StrictHostKeyChecking=accept-new -o WarnWeakCrypto=no -o PreferredAuthentications=password -o PubkeyAuthentication=no -o NumberOfPasswordPrompts=1 "$RemoteUser@$RemoteIP" $Command
}

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host " Starting Safe Sync to $RemoteIP (Port $RemotePort)..." -ForegroundColor Cyan
Write-Host " Local Path: $LocalPath"
Write-Host " Remote Path: $RemoteFullDir"
Write-Host "==========================================================" -ForegroundColor Cyan

if (-not (Test-Path $LocalPath)) {
    Write-Host "Error: local path not found: $LocalPath" -ForegroundColor Red
    exit 1
}

if (-not (Test-Path $RsyncExe)) {
    Write-Host "Error: rsync.exe not found at $RsyncExe" -ForegroundColor Red
    Write-Host "Please install MSYS2 rsync first." -ForegroundColor Red
    exit 1
}

if (-not (Test-Path $CygpathExe)) {
    Write-Host "Error: cygpath.exe not found at $CygpathExe" -ForegroundColor Red
    exit 1
}

if (-not (Test-Path $SshPassExe)) {
    Write-Host "Error: sshpass.exe not found at $SshPassExe" -ForegroundColor Red
    exit 1
}

if (-not (Test-Path $SshExe)) {
    Write-Host "Error: ssh.exe not found at $SshExe" -ForegroundColor Red
    exit 1
}

if (-not (Test-Path $SshPassFile)) {
    Write-Host "Error: SSH password file not found: $SshPassFile" -ForegroundColor Red
    exit 1
}

$env:Path = "$MsysBin;$env:Path"
$env:MSYS2_ARG_CONV_EXCL = "*"
$LocalPathPosix = (& $CygpathExe -u $LocalPath).Trim()
$SshPassFilePosix = (& $CygpathExe -u $SshPassFile).Trim()
$RemoteShell = "sshpass -f $SshPassFilePosix ssh -p $RemotePort -o StrictHostKeyChecking=accept-new -o WarnWeakCrypto=no -o PreferredAuthentications=password -o PubkeyAuthentication=no -o NumberOfPasswordPrompts=1"

Write-Host "[1/6] Checking remote rsync..." -ForegroundColor Yellow
$CheckRsyncCmd = "command -v rsync >/dev/null 2>&1"
Invoke-RemoteCommand $CheckRsyncCmd

if ($LASTEXITCODE -ne 0) {
    Write-Host "Error: rsync is not installed on the remote server, or password askpass login is not ready." -ForegroundColor Red
    exit 1
}

Write-Host "[2/6] Ensuring remote directory exists..." -ForegroundColor Yellow
$CreateDirCmd = "mkdir -p '$RemoteFullDir'"
Invoke-RemoteCommand $CreateDirCmd

if ($LASTEXITCODE -ne 0) {
    Write-Host "Error creating remote directory." -ForegroundColor Red
    exit $LASTEXITCODE
}

Write-Host "[3/6] Removing remote .git directory if present..." -ForegroundColor Yellow
$RemoveGitCmd = "if [ -d '$RemoteFullDir/.git' ]; then rm -rf '$RemoteFullDir/.git'; fi"
Invoke-RemoteCommand $RemoveGitCmd

if ($LASTEXITCODE -ne 0) {
    Write-Host "Warning: unable to remove remote .git directory. Continuing." -ForegroundColor Yellow
}

Write-Host "[4/6] Uploading local code with rsync..." -ForegroundColor Yellow
& $RsyncExe -av `
    --exclude=".git/" `
    --exclude="__pycache__/" `
    --exclude="*.pyc" `
    --exclude=".pytest_cache/" `
    --exclude=".ruff_cache/" `
    -e $RemoteShell `
    "$LocalPathPosix/" `
    $RemoteSpec

if ($LASTEXITCODE -ne 0) {
    Write-Host "Error during rsync upload." -ForegroundColor Red
    exit $LASTEXITCODE
}

Write-Host "==========================================================" -ForegroundColor Green
Write-Host " Upload Complete! Now setting permissions..." -ForegroundColor Green
Write-Host "==========================================================" -ForegroundColor Green

Write-Host "[5/6] Setting permissions (chmod 755)..." -ForegroundColor Yellow
$ChmodCmd = "chmod -R 755 '$RemoteFullDir'"
Invoke-RemoteCommand $ChmodCmd

if ($LASTEXITCODE -ne 0) {
    Write-Host "Error setting permissions." -ForegroundColor Red
}

Write-Host "[6/6] Fixing line endings for .sbatch and .sh files..." -ForegroundColor Yellow
$Dos2UnixCmd = "find '$RemoteFullDir' -type f \( -name '*.sbatch' -o -name '*.sh' \) -exec sed -i 's/\r$//' {} +"
Invoke-RemoteCommand $Dos2UnixCmd

if ($LASTEXITCODE -eq 0) {
    Write-Host " Sync, Permissions, and Line Endings Set Successfully!" -ForegroundColor Green
}
else {
    Write-Host "Error fixing line endings." -ForegroundColor Red
}
