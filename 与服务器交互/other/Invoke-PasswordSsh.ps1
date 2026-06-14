<#
.SYNOPSIS
通过 Windows OpenSSH 的 SSH_ASKPASS 机制执行密码登录，不在脚本中保存密码。

.DESCRIPTION
适用于公钥未被远端接受、但用户明确提供了密码授权的非交互任务。
调用者需先在当前 PowerShell 进程设置 `CODEX_SSH_PASSWORD`（或由
`-PasswordEnvVar` 指定的变量）；脚本只创建不含密码的一次性 askpass helper，
运行结束后自动删除临时文件。

.EXAMPLE
$env:CODEX_SSH_PASSWORD = '<password provided by user>'
& '<project>\与服务器交互\other\Invoke-PasswordSsh.ps1' `
  -HostName 'server.example' -Port 22 -UserName 'user' -Command 'hostname'
Remove-Item Env:CODEX_SSH_PASSWORD

.EXAMPLE
$env:CODEX_SSH_PASSWORD = '<password provided by user>'
& '<project>\与服务器交互\other\Invoke-PasswordSsh.ps1' `
  -HostName 'server.example' -Port 22 -UserName 'user' `
  -Command 'bash -s' -InputFile 'C:\path\probe.sh'
Remove-Item Env:CODEX_SSH_PASSWORD
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$HostName,

    [Parameter(Mandatory = $true)]
    [string]$UserName,

    [int]$Port = 22,

    [string]$Command = "hostname",

    [string]$InputFile,

    [string]$PasswordEnvVar = "CODEX_SSH_PASSWORD",

    [switch]$SkipHostKeyCheck
)

$password = [Environment]::GetEnvironmentVariable($PasswordEnvVar, "Process")
if ([string]::IsNullOrEmpty($password)) {
    throw "Set process environment variable '$PasswordEnvVar' before invoking this script."
}

$tempRoot = Join-Path ([System.IO.Path]::GetTempPath()) ("codex-ssh-askpass-" + [Guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Path $tempRoot | Out-Null
$askPassPs1 = Join-Path $tempRoot "askpass.ps1"
$askPassCmd = Join-Path $tempRoot "askpass.cmd"
$emptyInput = Join-Path $tempRoot "stdin.txt"

try {
    @'
[Console]::Out.Write($env:CODEX_SSH_PASSWORD)
'@ | Set-Content -LiteralPath $askPassPs1 -Encoding ASCII
    "@echo off`r`npowershell.exe -NoProfile -NonInteractive -ExecutionPolicy Bypass -File `"$askPassPs1`"`r`n" |
        Set-Content -LiteralPath $askPassCmd -Encoding ASCII
    Set-Content -LiteralPath $emptyInput -Value "" -Encoding ASCII

    $previousAskPass = $env:SSH_ASKPASS
    $previousAskPassRequire = $env:SSH_ASKPASS_REQUIRE
    $previousDisplay = $env:DISPLAY
    $previousPassword = $env:CODEX_SSH_PASSWORD
    $env:CODEX_SSH_PASSWORD = $password
    $env:SSH_ASKPASS = $askPassCmd
    $env:SSH_ASKPASS_REQUIRE = "force"
    $env:DISPLAY = "codex"

    $options = @(
        "-o", "PubkeyAuthentication=no",
        "-o", "PreferredAuthentications=password",
        "-o", "NumberOfPasswordPrompts=1",
        "-o", "ConnectTimeout=10",
        "-p", $Port.ToString()
    )
    if ($SkipHostKeyCheck) {
        $options += @("-o", "StrictHostKeyChecking=no")
    } else {
        $options += @("-o", "StrictHostKeyChecking=yes")
    }
    $options += @("$UserName@$HostName", $Command)
    $quotedOptions = $options | ForEach-Object { '"' + ($_ -replace '"', '\"') + '"' }
    $inputPath = if ($InputFile) { (Resolve-Path -LiteralPath $InputFile).Path } else { $emptyInput }

    $process = Start-Process -FilePath "ssh.exe" -ArgumentList ($quotedOptions -join " ") -NoNewWindow -Wait -PassThru `
        -RedirectStandardInput $inputPath `
        -RedirectStandardOutput (Join-Path $tempRoot "stdout.txt") `
        -RedirectStandardError (Join-Path $tempRoot "stderr.txt")
    Get-Content -LiteralPath (Join-Path $tempRoot "stdout.txt") -Raw
    $stderr = Get-Content -LiteralPath (Join-Path $tempRoot "stderr.txt") -Raw
    if ($stderr) {
        [Console]::Error.Write($stderr)
    }
    if ($process.ExitCode -ne 0) {
        exit $process.ExitCode
    }
} finally {
    $env:SSH_ASKPASS = $previousAskPass
    $env:SSH_ASKPASS_REQUIRE = $previousAskPassRequire
    $env:DISPLAY = $previousDisplay
    $env:CODEX_SSH_PASSWORD = $previousPassword
    Remove-Item -LiteralPath $tempRoot -Recurse -Force -ErrorAction SilentlyContinue
}
