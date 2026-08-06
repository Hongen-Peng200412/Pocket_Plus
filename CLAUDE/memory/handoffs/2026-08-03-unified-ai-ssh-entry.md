# Handoff：Windows 统一 AI SSH 入口

Date: 2026-08-03

## Current State

Windows 用户级统一入口已经建立，Pocket_Plus 的项目入口只负责把参数转交给它。项目文件和交接记录均不保存服务器密码。

统一入口为 `C:\Users\15919\.codex\tools\Invoke-ProjectSsh.ps1`，Pocket_Plus 入口为 `与服务器交互/other/Invoke-PasswordSsh.ps1`。AI agent 可以通过 `-Command` 执行单条远端命令，也可以通过 `-Command "bash -s" -InputFile <LF 脚本>` 执行本地脚本。

密码保存在 `C:\Users\15919\.ssh\pocket_plus_sshpass.txt`。该文件已关闭继承，只允许当前 Windows 用户、SYSTEM 和本机 Administrators 访问。服务器主机密钥继续使用 `C:\Users\15919\.ssh\known_hosts` 中 `[10.102.33.220]:10022` 的固定记录；入口禁止跳过主机密钥校验。

## Completed

- 新建用户级统一 SSH 入口，集中管理服务器地址、端口、账号、密码读取、固定主机密钥校验和有限次数重试。
- Pocket_Plus 的旧入口改为薄包装，不再要求调用者显式提供密码文件。
- 更新 `与服务器交互/other/readme.md`，补充命令执行和 LF 脚本执行示例。
- 更新全局 `project-server-interaction` skill，使其他 AI agent 优先发现并使用同一入口。
- PowerShell 语法检查通过；全局 skill 校验通过。
- 已验证连接失败会返回非零退出状态，不会把失败误报为成功。
- 关闭 ATrust 后已完成真实登录验收：服务器返回主机名 `master` 和账号 `penghongen`，两个项目目录与 `/usr/bin/sbatch` 均可访问。
- 已从 AdaLigand 入口通过标准输入执行 LF 脚本，并在服务器上完成 AdaLigand、Pocket_Plus 两份 `submit_task.sh` 的 `bash -n` 检查；Pocket_Plus 项目入口也已独立登录成功。

## Decisions

- 统一入口位于 Windows 用户目录，不归属于单个代码库，因此 AdaLigand、Pocket_Plus 和其他本机项目可以复用。
- 密码只保存在本机私有文件，不写入 Git 仓库、日志、交接记录或公开网站。
- 只对发生在身份验证之前的临时断连进行有限次数重试；认证失败和主机密钥不匹配立即失败。
- 保留严格主机密钥校验，不提供关闭校验的参数。
- 本轮不提交真实 Slurm 任务；入口只提供可靠的命令和脚本执行能力。

## Open Questions

- 无。ATrust 开启时，新 SSH 连接曾在协议横幅前被中断；关闭 ATrust 后，`run_sync.bat` 与统一 AI SSH 入口同时恢复。MobaXterm 当时存在已建立连接，但现有证据不支持“它占满连接配额”的旧推断。

## Next Actions

1. 后续 AI agent 直接调用统一入口或任一项目薄入口。
2. 若再次出现认证前断连，先检查 ATrust 或其他 VPN 的私网路由，再使用入口的有限重试。
3. 真实计算仍通过项目正式 Slurm 入口提交；没有新的明确授权时不提交计算任务。

## Files To Reopen

- `C:\Users\15919\.codex\tools\Invoke-ProjectSsh.ps1`
- `C:\Users\15919\.ssh\pocket_plus_sshpass.txt`
- `与服务器交互/other/Invoke-PasswordSsh.ps1`
- `与服务器交互/other/readme.md`
- `C:\Users\15919\.codex\skills\project-server-interaction\SKILL.md`
