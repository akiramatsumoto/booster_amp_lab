#!/usr/bin/env bash
# ML3 から手元のノートPC (Tailscale 経由) へ ONNX を送る。
#
# 使い方:
#   ./scripts/send_onnx.sh                                # 全タスクで一番新しい .onnx を送る
#   ./scripts/send_onnx.sh soccer_stand_kick_amp          # そのタスクの最新 run の .onnx
#   ./scripts/send_onnx.sh soccer_stand_kick_amp -r 2026-07-13_12-40-54   # run 指定
#   ./scripts/send_onnx.sh --list                         # 送らずに候補だけ表示
#   ./scripts/send_onnx.sh --cmd                          # ノートPC 側で叩く scp コマンドを表示
#   ./scripts/send_onnx.sh -H akira-1 -d some/other/dir soccer_stand_kick_amp
#
# 接続先/転送先は環境変数でも指定できる:
#   ONNX_SEND_HOST  既定 akira-1  (Tailscale のマシン名。100.82.182.36 でも可)
#   ONNX_SEND_DEST  既定 ~/futbol_main/src/ros2_ws/src/booster_k1_locomotion/assets
#
# 注意: Tailscale SSH の ACL が check モードのため、初回接続時にブラウザでの承認を
# 求められて止まる (URL が表示されるので開いて承認する)。承認すると checkPeriod の
# 間は無言で通る。cron 等で完全に無人実行したいなら、管理コンソールで該当 ACL を
# check → accept に変えるか、--cmd でノートPC 側から引っ張る。
set -euo pipefail

LOCAL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_ROOT="$LOCAL_ROOT/logs/rsl_rl"

HOST="${ONNX_SEND_HOST:-akira-1}"
DEST="${ONNX_SEND_DEST:-~/futbol_main/src/ros2_ws/src/booster_k1_locomotion/assets}"
TASK=""
RUN=""
LIST=0
CMD_ONLY=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    -H|--host) HOST="$2"; shift 2 ;;
    -d|--dest) DEST="$2"; shift 2 ;;
    -r|--run)  RUN="$2";  shift 2 ;;
    --list|-l) LIST=1; shift ;;
    --cmd|-c)  CMD_ONLY=1; shift ;;
    -h|--help) sed -n '2,20p' "${BASH_SOURCE[0]}"; exit 0 ;;
    -*) echo "unknown option: $1" >&2; exit 1 ;;
    *)  TASK="$1"; shift ;;
  esac
done

# ML3 側の自分の identity (ノートPC から引っ張るときの接続先)。
# MagicDNS が不調なことがあるので Tailscale IP を優先し、駄目なら hostname に落とす。
MY_IP="$(tailscale ip -4 2>/dev/null | head -1 || true)"
if [[ -n "$MY_IP" ]]; then
  ME="$(whoami)@$MY_IP"
else
  ME="$(whoami)@$(hostname)"
fi

# --- 送る .onnx を決める ------------------------------------------------------
if [[ -n "$RUN" ]]; then
  [[ -n "$TASK" ]] || { echo "-r/--run を使うときはタスク名も指定してください。" >&2; exit 1; }
  SEARCH_DIR="$LOG_ROOT/$TASK/$RUN/exported"
elif [[ -n "$TASK" ]]; then
  SEARCH_DIR="$LOG_ROOT/$TASK"
else
  SEARCH_DIR="$LOG_ROOT"
fi

[[ -d "$SEARCH_DIR" ]] || { echo "$SEARCH_DIR がありません" >&2; exit 2; }

# 更新時刻が一番新しい .onnx を 1 つ選ぶ
ONNX="$(find "$SEARCH_DIR" -name '*.onnx' -printf '%T@ %p\n' 2>/dev/null \
        | sort -nr | head -1 | cut -d' ' -f2-)"

[[ -n "$ONNX" ]] || { echo "$SEARCH_DIR 以下に .onnx がありません" >&2; exit 3; }

echo "onnx : ${ONNX#$LOCAL_ROOT/}"
echo "size : $(du -h "$ONNX" | cut -f1)"
echo "mtime: $(date -r "$ONNX" '+%Y-%m-%d %H:%M:%S')"

if [[ "$LIST" -eq 1 ]]; then
  echo
  echo "他の候補 (新しい順):"
  find "$SEARCH_DIR" -name '*.onnx' -printf '%T@ %p\n' 2>/dev/null \
    | sort -nr | head -10 | cut -d' ' -f2- | sed "s#^$LOCAL_ROOT/#  #"
  exit 0
fi

# ノートPC 側で叩く「引っ張る」コマンド
pull_cmd() {
  echo "scp $ME:$ONNX $DEST/"
}

if [[ "$CMD_ONLY" -eq 1 ]]; then
  echo
  echo "ノートPC 側で以下を実行してください:"
  echo "  $(pull_cmd)"
  exit 0
fi

# --- 転送 ---------------------------------------------------------------------
# BatchMode は付けない: Tailscale SSH の check (ブラウザ承認) を通す必要があるため。
echo "→ $HOST:$DEST/"
echo "(初回はブラウザ承認の URL が出ることがあります)"
echo

SSH_OPTS=(-o StrictHostKeyChecking=accept-new)

if ! ssh "${SSH_OPTS[@]}" -n "$HOST" "mkdir -p $DEST"; then
  echo
  echo "$HOST に SSH で入れませんでした。" >&2
  echo "ノートPC 側で以下を実行して引っ張る方法もあります:" >&2
  echo >&2
  echo "  $(pull_cmd)" >&2
  exit 4
fi

scp "${SSH_OPTS[@]}" "$ONNX" "$HOST:$DEST/"

echo
echo "送信完了: $(basename "$ONNX")"
echo "  → $HOST:$DEST/$(basename "$ONNX")"
