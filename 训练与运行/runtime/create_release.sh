#!/usr/bin/env bash
set -euo pipefail

# 冻结项目目录并按内容复用 release。调用者只读取 stdout 的最后结果：
#   <release 目录>/<项目目录名>
# 过程信息全部写入 stderr，避免命令替换得到混杂路径。

fail() {
    printf '[create_release][错误] %s\n' "$*" >&2
    exit 2
}

[[ $# -eq 2 ]] || fail "用法：create_release.sh TASK_ROOT RELEASES_ROOT"
source_root="$(cd "$1" && pwd -P)"
releases_root="$2"
project_name="$(basename "${source_root}")"
mkdir -p "${releases_root}"
releases_root="$(cd "${releases_root}" && pwd -P)"

command -v rsync >/dev/null 2>&1 || fail "缺少 rsync"
command -v sha256sum >/dev/null 2>&1 || fail "缺少 sha256sum"
command -v stat >/dev/null 2>&1 || fail "缺少 stat"

compute_tree_hash() {
    local tree_root="$1"
    (
        cd "${tree_root}"
        find . \
            \( -type d \( -name .git -o -name __pycache__ -o -name .pytest_cache \) -prune \) \
            -o \( \( -type f -o -type l \) ! -name '*.pyc' -print0 \) |
            sort -z |
            while IFS= read -r -d '' relative_path; do
                if [[ -L "${relative_path}" ]]; then
                    printf 'L\t%s\t%s\0' "${relative_path}" "$(readlink "${relative_path}")"
                else
                    printf 'F\t%s\t%s\t%s\t' \
                        "${relative_path}" \
                        "$(stat -c '%a' "${relative_path}")" \
                        "$(stat -c '%s' "${relative_path}")"
                    sha256sum "${relative_path}" | awk '{printf "%s%c", $1, 0}'
                fi
            done
    ) | sha256sum | awk '{print $1}'
}

json_escape() {
    sed -e 's/\\/\\\\/g' -e 's/"/\\"/g'
}

content_hash="$(compute_tree_hash "${source_root}")"
release_id="${project_name}_$(printf '%s' "${content_hash}" | cut -c1-12)"
release_directory="${releases_root}/${release_id}"
release_project_root="${release_directory}/${project_name}"

if [[ -d "${release_project_root}" ]]; then
    existing_hash="$(sed -n 's/.*"content_sha256": *"\([^"]*\)".*/\1/p' \
        "${release_directory}/manifest.json" 2>/dev/null || true)"
    [[ "${existing_hash}" == "${content_hash}" ]] \
        || fail "已有 release 目录的内容身份不匹配：${release_directory}"
    printf '[create_release] 复用 %s\n' "${release_directory}" >&2
    printf '%s\n' "${release_project_root}"
    exit 0
fi

temporary_directory="${releases_root}/.${release_id}.tmp.$$"
case "${temporary_directory}" in
    "${releases_root}/."*) ;;
    *) fail "临时 release 路径越界：${temporary_directory}" ;;
esac
cleanup_temporary() {
    if [[ -d "${temporary_directory}" ]]; then
        rm -rf -- "${temporary_directory}"
    fi
}
trap cleanup_temporary EXIT

mkdir -p "${temporary_directory}/${project_name}"
rsync -a \
    --exclude '.git/' \
    --exclude '__pycache__/' \
    --exclude '.pytest_cache/' \
    --exclude '*.pyc' \
    "${source_root}/" "${temporary_directory}/${project_name}/"

copied_hash="$(compute_tree_hash "${temporary_directory}/${project_name}")"
[[ "${copied_hash}" == "${content_hash}" ]] \
    || fail "release 写后内容哈希不一致"

git_commit=""
git_branch=""
git_dirty=""
if git -C "${source_root}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    git_commit="$(git -C "${source_root}" rev-parse HEAD 2>/dev/null || true)"
    git_branch="$(git -C "${source_root}" rev-parse --abbrev-ref HEAD 2>/dev/null || true)"
    if [[ -n "$(git -C "${source_root}" status --porcelain 2>/dev/null || true)" ]]; then
        git_dirty=true
    else
        git_dirty=false
    fi
fi

created_at="$(date --iso-8601=seconds)"
cat >"${temporary_directory}/manifest.json" <<EOF
{
  "schema_version": 1,
  "release_id": "$(printf '%s' "${release_id}" | json_escape)",
  "project_name": "$(printf '%s' "${project_name}" | json_escape)",
  "source_root": "$(printf '%s' "${source_root}" | json_escape)",
  "created_at": "$(printf '%s' "${created_at}" | json_escape)",
  "content_sha256": "${content_hash}",
  "excluded": [".git/", "__pycache__/", ".pytest_cache/", "*.pyc"],
  "git_commit": "$(printf '%s' "${git_commit}" | json_escape)",
  "git_branch": "$(printf '%s' "${git_branch}" | json_escape)",
  "git_dirty": "${git_dirty}"
}
EOF

if [[ -e "${release_directory}" ]]; then
    cleanup_temporary
else
    mv "${temporary_directory}" "${release_directory}"
fi
trap - EXIT

printf '[create_release] 创建 %s\n' "${release_directory}" >&2
printf '%s\n' "${release_project_root}"
