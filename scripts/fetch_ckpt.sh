#!/usr/bin/env bash
# vast.ai のインスタンスから最新 run の最新チェックポイントを取ってくる。
#
# 使い方:
#   VAST_SSH="-p 40712 root@ssh4.vast.ai" ./scripts/fetch_ckpt.sh soccer_stand_kick_amp
#   ./scripts/fetch_ckpt.sh -H ssh4.vast.ai -P 40712 soccer_stand_kick_amp
#   ./scripts/fetch_ckpt.sh --list soccer_stand_kick_amp     # 取ってこずに候補だけ表示
#   ./scripts/fetch_ckpt.sh --full soccer_stand_kick_amp     # run ディレクトリ丸ごと
#
# run ディレクトリは YYYY-MM-DD_HH-MM-SS 形式のものだけを対象にする
# (10525 のような非日付ディレクトリは無視される)。
set -euo pipefail

REMOTE_ROOT="${VAST_REMOTE_ROOT:-/workspace/booster_amp_lab}"
LOCAL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

TASK=""
FULL=0
LIST=0
HOST="${VAST_HOST:-}"
PORT="${VAST_PORT:-}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    -H|--host) HOST="$2"; shift 2 ;;
    -P|--port) PORT="$2"; shift 2 ;;
    --full)    FULL=1; shift ;;
    --list|-l) LIST=1; shift ;;
    -h|--help) sed -n '2,11p' "${BASH_SOURCE[0]}"; exit 0 ;;
    -*) echo "unknown option: $1" >&2; exit 1 ;;
    *)  TASK="$1"; shift ;;
  esac
done

# --- ssh 接続先の組み立て -----------------------------------------------------
if [[ -n "${VAST_SSH:-}" ]]; then
  # 例: VAST_SSH="-p 40712 root@ssh4.vast.ai"  (vast.ai が出す ssh コマンドの引数部分)
  read -r -a _args <<<"$VAST_SSH"
  SSH_TARGET="${_args[-1]}"
  SSH_OPTS=("${_args[@]:0:${#_args[@]}-1}")
  SCP_OPTS=()
  for a in "${SSH_OPTS[@]}"; do
    if [[ "$a" == "-p" ]]; then SCP_OPTS+=("-P"); else SCP_OPTS+=("$a"); fi
  done
elif [[ -n "$HOST" ]]; then
  if [[ "$HOST" == *"@"* ]]; then SSH_TARGET="$HOST"; else SSH_TARGET="root@$HOST"; fi
  SSH_OPTS=(); SCP_OPTS=()
  if [[ -n "$PORT" ]]; then SSH_OPTS=(-p "$PORT"); SCP_OPTS=(-P "$PORT"); fi
else
  echo "ssh 接続先が未指定です。VAST_SSH か -H/-P を指定してください。" >&2
  echo '  例: VAST_SSH="-p 40712 root@ssh4.vast.ai" ./scripts/fetch_ckpt.sh soccer_stand_kick_amp' >&2
  exit 1
fi

SSH=(ssh -o StrictHostKeyChecking=accept-new "${SSH_OPTS[@]}" "$SSH_TARGET")
SCP=(scp -o StrictHostKeyChecking=accept-new "${SCP_OPTS[@]}")

# --- タスク名が無ければリモートの一覧を出して終わる ---------------------------
if [[ -z "$TASK" ]]; then
  echo "タスク名を指定してください。リモートにあるのは:" >&2
  "${SSH[@]}" "ls -1 '$REMOTE_ROOT/logs/rsl_rl' 2>/dev/null" >&2 || true
  exit 1
fi

REMOTE_TASK_DIR="$REMOTE_ROOT/logs/rsl_rl/$TASK"

# --- 最新 run と最大 step を解決 ----------------------------------------------
read -r RUN STEP < <("${SSH[@]}" bash -s <<EOF
set -e
cd "$REMOTE_TASK_DIR" 2>/dev/null || { echo "リモートに $REMOTE_TASK_DIR がありません" >&2; exit 2; }
run=\$(ls -1d */ 2>/dev/null | tr -d '/' \
      | grep -E '^[0-9]{4}-[0-9]{2}-[0-9]{2}_[0-9]{2}-[0-9]{2}-[0-9]{2}\$' \
      | sort | tail -1)
[ -n "\$run" ] || { echo "日付形式の run ディレクトリが見つかりません" >&2; exit 3; }
step=\$(ls -1 "\$run"/model_*.pt 2>/dev/null \
       | sed -e 's#.*/model_##' -e 's#\.pt\$##' | sort -n | tail -1)
[ -n "\$step" ] || { echo "\$run に model_*.pt がありません" >&2; exit 4; }
echo "\$run \$step"
EOF
)

echo "task : $TASK"
echo "run  : $RUN"
echo "ckpt : model_${STEP}.pt"

[[ "$LIST" -eq 1 ]] && exit 0

# --- 転送 ---------------------------------------------------------------------
LOCAL_RUN_DIR="$LOCAL_ROOT/logs/rsl_rl/$TASK/$RUN"
mkdir -p "$LOCAL_RUN_DIR"

if [[ "$FULL" -eq 1 ]]; then
  "${SCP[@]}" -r "$SSH_TARGET:$REMOTE_TASK_DIR/$RUN/." "$LOCAL_RUN_DIR/"
else
  # play に必要なのは checkpoint と params/ (env.yaml, agent.yaml)
  "${SCP[@]}" "$SSH_TARGET:$REMOTE_TASK_DIR/$RUN/model_${STEP}.pt" "$LOCAL_RUN_DIR/"
  "${SCP[@]}" -r "$SSH_TARGET:$REMOTE_TASK_DIR/$RUN/params" "$LOCAL_RUN_DIR/" 2>/dev/null || true
fi

echo
echo "→ $LOCAL_RUN_DIR"
ls -la "$LOCAL_RUN_DIR"
