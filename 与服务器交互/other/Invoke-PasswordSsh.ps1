<#
.SYNOPSIS
调用本机统一 SSH 入口执行轻量远程命令。

.DESCRIPTION
项目只保留稳定入口；密码读取、严格主机密钥检查、有限退避重试和退出码传播由
%USERPROFILE%\.codex\tools\Invoke-ProjectSsh.ps1 统一实现。调用者不需要把密码写入命令。

.EXAMPLE
& ".\与服务器交互\other\Invoke-PasswordSsh.ps1" -Command "hostname"

.EXAMPLE
& ".\与服务器交互\other\Invoke-PasswordSsh.ps1" `
  -Command "bash -s" -InputFile ".\tmp\probe.sh"
#>
[CmdletBinding()]
param(
    [string]$HostName,
    [int]$Port,
    [string]$UserName,
    [string]$Command,
    [string]$InputFile,
    [string]$PasswordEnvVar,
    [string]$PasswordFile,
    [int]$RetryCount,
    [int]$RetryDelaySeconds,
    [switch]$SkipHostKeyCheck
)

$sharedEntry = Join-Path $env:USERPROFILE ".codex\tools\Invoke-ProjectSsh.ps1"
if (-not (Test-Path -LiteralPath $sharedEntry -PathType Leaf)) {
    throw "本机统一 SSH 入口不存在：$sharedEntry"
}

& $sharedEntry @PSBoundParameters
