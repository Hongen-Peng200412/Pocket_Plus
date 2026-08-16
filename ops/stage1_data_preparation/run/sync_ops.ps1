<#
.SYNOPSIS
只把当前工作树的 Stage1 数据准备目录安全同步到 Pocket Plus 服务器项目。

.DESCRIPTION
本脚本不删除远端文件，不上传仓库其他目录，也不接受未知主机密钥。它从当前脚本
位置解析工作树和来源目录，使用本机私有密码文件完成 rsync，并在远端只修正本目录
内 shell 文件的执行权限。
#>

[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$ScriptDirectory = Split-Path -Parent $MyInvocation.MyCommand.Path
$SourceDirectory = (Resolve-Path (Join-Path $ScriptDirectory "..")).Path
$RemoteDirectory = "/home/penghongen/My_Project/Pocket_Plus/ops/stage1_data_preparation"
$PasswordFile = Join-Path $env:USERPROFILE ".ssh\pocket_plus_sshpass.txt"
$MsysRoots = @($env:PROJECT_MSYS_ROOT, $env:MSYS2_ROOT, "C:\msys64", "D:\msys64") |
    Where-Object { -not [string]::IsNullOrWhiteSpace($_) }

$MsysBin = $null
foreach ($Root in $MsysRoots) {
    $ExpandedRoot = [Environment]::ExpandEnvironmentVariables($Root.Trim())
    $Candidate = if ($ExpandedRoot.EndsWith("\usr\bin")) {
        $ExpandedRoot
    }
    else {
        Join-Path $ExpandedRoot "usr\bin"
    }
    $RequiredNames = @("rsync.exe", "cygpath.exe", "sshpass.exe", "ssh.exe")
    $MissingNames = $RequiredNames | Where-Object {
        -not (Test-Path -LiteralPath (Join-Path $Candidate $_) -PathType Leaf)
    }
    if (-not $MissingNames) {
        $MsysBin = $Candidate
        break
    }
}
if (-not $MsysBin) {
    throw "未找到包含 rsync、cygpath、sshpass 和 ssh 的 MSYS2 工具目录。"
}
if (-not (Test-Path -LiteralPath $PasswordFile -PathType Leaf)) {
    throw "本机私有密码文件不存在：$PasswordFile"
}

$RsyncExecutable = Join-Path $MsysBin "rsync.exe"
$CygpathExecutable = Join-Path $MsysBin "cygpath.exe"
$SourcePosix = (& $CygpathExecutable -u $SourceDirectory).Trim()
$PasswordPosix = (& $CygpathExecutable -u $PasswordFile).Trim()
$RemoteShell = "sshpass -f $PasswordPosix ssh -p 10022 -o StrictHostKeyChecking=yes -o PreferredAuthentications=password -o PubkeyAuthentication=no -o NumberOfPasswordPrompts=1"

$env:MSYS2_ARG_CONV_EXCL = "*"
& $RsyncExecutable -av `
    --exclude="__pycache__/" `
    --exclude="*.pyc" `
    --exclude=".pytest_cache/" `
    -e $RemoteShell `
    "$SourcePosix/" `
    "penghongen@10.102.33.220:${RemoteDirectory}/"
if ($LASTEXITCODE -ne 0) {
    throw "Stage1 数据准备目录同步失败，rsync 退出码为 $LASTEXITCODE。"
}

$SshPassExecutable = Join-Path $MsysBin "sshpass.exe"
$SshExecutable = Join-Path $MsysBin "ssh.exe"
& $SshPassExecutable -f $PasswordPosix $SshExecutable `
    -p 10022 `
    -o StrictHostKeyChecking=yes `
    -o PreferredAuthentications=password `
    -o PubkeyAuthentication=no `
    -o NumberOfPasswordPrompts=1 `
    "penghongen@10.102.33.220" `
    "find '$RemoteDirectory/run' -type f -name '*.sh' -exec chmod 755 {} +"
if ($LASTEXITCODE -ne 0) {
    throw "远端 shell 执行权限设置失败，SSH 退出码为 $LASTEXITCODE。"
}
