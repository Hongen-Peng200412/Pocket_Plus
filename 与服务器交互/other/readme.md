# Pocket Plus 服务器交互工具箱

本目录记录 Pocket Plus 项目的服务器交互工具与使用纪律。工具随项目走，密码、本机依赖和 VS Code 用户设置不随项目走。

## 布局

`与服务器交互` 根目录保留常用入口：

- `run_sync.bat`：安全同步入口，用户可双击；AI agent 只有在用户明确要求同步时才可运行。
- `sync_code.ps1`：安全同步实现，只上传本地项目到远端目录，不删除远端项目目录。
- `run_syncWithClean.bat`：删除式同步入口，人类手动专用。
- `sync_codeWithClean.ps1`：删除式同步实现，会先删除远端目标目录，AI agent 禁止擅自运行。

`与服务器交互\other` 放辅助工具和说明：

- `Invoke-PasswordSsh.ps1`：AI agent 轻量远端命令 helper。
- `readme.md`：本说明。

`与服务器交互\sbatch` 放 Pocket Plus 的 Slurm 调度模板或项目适配脚本。当前模板机制可作为其他项目参考，但模板中的 Pocket Plus 环境名、路径和入口命令不能直接假定适用于其他项目。

## 当前映射

- 本地项目：`...(取决于设备)\OneDrive\My_Project\Pocket_Plus`
- 远端目录：`/home/penghongen/My_Project/Pocket_Plus`
- 服务器：`penghongen@10.102.33.220:10022`

## 免密推荐方案

用户视角下的一切免密，推荐分工具实现，而不是强行用同一种机制：

- 同步入口免密：使用 MSYS2 `rsync + sshpass + ssh`，密码存放在本机私有文件 `%USERPROFILE%\.ssh\pocket_plus_sshpass.txt`，项目内不保存密码。
- VS Code 免密：使用 `%USERPROFILE%\.ssh\ssh_with_askpass.cmd` 作为 `remote.SSH.path`，并在 SSH config 中配置 `emap-server`。
- AI 远端探测免密：由 agent 在当前进程临时设置 `CODEX_SSH_PASSWORD`，再调用 `other\Invoke-PasswordSsh.ps1`；该脚本不保存密码。

服务器当前不提供 `publickey` 认证方式，因此 SSH key 免密不是当前可用方案。若以后服务器开放 publickey，应优先切回 SSH key。

## 依赖

同步脚本依赖 MSYS2 工具链：

- `rsync.exe`
- `sshpass.exe`
- `ssh.exe`
- `cygpath.exe`

查找顺序是有限的；候选目录或盘符不存在时应跳过，不应报错中断：

1. 环境变量 `PROJECT_MSYS_ROOT`。
2. 环境变量 `MSYS2_ROOT`。
3. `C:\msys64`。
4. `D:\msys64`。

同步脚本必须设置 `MSYS2_ARG_CONV_EXCL=*`，否则 MSYS2 可能把远端 `/home/...` 路径改写成 `C:/msys64/home/...`。

## AI helper 使用纪律

`Invoke-PasswordSsh.ps1` 用于轻量远端命令、只读探测或把本地 LF 行尾的 bash 脚本通过 stdin 送给远端 `bash -s`。

调用前由当前 PowerShell 进程设置密码环境变量：

```powershell
$env:CODEX_SSH_PASSWORD = "<user-provided password>"
& "<project>\与服务器交互\other\Invoke-PasswordSsh.ps1" `
  -HostName "10.102.33.220" -Port 10022 -UserName "penghongen" `
  -Command "hostname"
Remove-Item Env:\CODEX_SSH_PASSWORD
```

注意：

- 不把密码写入项目文件。
- 不用它跑模型训练、模型加载或重型推理。
- 远端写入默认只允许在用户明确授权的位置进行。
- `-InputFile` 传给远端 bash 时必须使用 LF 行尾。

## sbatch 与资源纪律

- `try_lock_X`：只有用户明确说“使用 `try_lock_X` 对应的资源”时，AI agent 才可围绕该 job 写入 `run_cmd_X.sh` 并触发执行。
- `pre_lock_X`：只有用户明确说“使用 `pre_lock_X` 对应的资源”时，AI agent 才可围绕该 job 写入 `run_cmd_X.sh` 并触发执行。
- `after_lock_X`：除非用户明确要求释放资源，不删除。
- `kill_lock_X`：除非用户明确授权使用或明确要求终止对应资源，不创建、不删除、不触碰。

普通命令行会话不跑模型训练、模型加载或重型推理；这类任务应进入 sbatch/lock 资源流程。

## 跨设备复用条件

可复用部分：

- 项目级 `与服务器交互` 布局。
- `other\Invoke-PasswordSsh.ps1`。
- safe/clean 同步纪律。
- `sbatch` 模板机制和 lock 资源纪律。

每台设备需要自行准备：

- MSYS2 与 `rsync/sshpass/ssh/cygpath`。
- 本机私有密码文件或等效安全凭据。
- VS Code 用户设置与 SSH config。
- 该设备自己的本地项目路径。
