#!/bin/sh
# 容器入口：准备依赖 → 启动 Streamlit。
# 支持的两种用法：
#   1) Dockerfile 已把依赖装进 /opt/venv：跳过安装直接启动；
#   2) 直接用 python:3.13-slim 镜像 + 挂载源码：首次启动时把依赖装到 /opt/venv 卷里。
# 镜像的 Python 版本需要与生成索引/复算记录的版本一致（本机为 3.13 → Unicode 15.1.0，
# 该版本号写在记录里，版本不同会报“复算不一致：$.index.identity.unicode_version”）。
# 状态标记写到 $STATUS_DIR（默认 /app/status，挂在宿主目录上），便于排查。
set -eu

APP_DIR=${APP_DIR:-/app}
VENV=${VENV:-/opt/venv}
DATA_DIR=${NLP_DATA_DIR:-/app/.data}
STATUS_DIR=${STATUS_DIR:-/app/status}
PIP_MIRROR=${PIP_MIRROR:-https://pypi.tuna.tsinghua.edu.cn/simple}
PORT=${STREAMLIT_PORT:-8501}

export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export NLP_DATA_DIR="$DATA_DIR"

cd "$APP_DIR" 2>/dev/null || true

mkdir -p "$DATA_DIR/conversations" "$DATA_DIR/indexes" "$DATA_DIR/traces"

if [ -d "$STATUS_DIR" ]; then
    rm -f "$STATUS_DIR"/0*-* 2>/dev/null || true
    mark() { : > "$STATUS_DIR/$1"; }
    LOG="$STATUS_DIR/deploy.log"
else
    mark() { :; }
    LOG=/dev/null
fi

echo "[nlp] 启动中：$(date -u '+%Y-%m-%d %H:%M:%S UTC')"
mark 01-entrypoint-started

# 解释器版本与 Unicode 数据版本都会进入索引标识（identity.unicode_version），
# 换镜像 tag 会导致旧索引/旧复算记录无法复算，因此把这两项一起记进环境标记。
IMG_VER=$(python -c 'import sys; print("%d.%d" % sys.version_info[:2])')
IMG_UNI=$(python -c 'import unicodedata; print(unicodedata.unidata_version)')
REQ_HASH=$(sha256sum "$APP_DIR/requirements.txt" | cut -d' ' -f1)
TARGET="$IMG_VER|$IMG_UNI|$REQ_HASH"
STATE="$VENV/.nlp-env"
echo "[nlp] 目标环境：Python $IMG_VER / Unicode $IMG_UNI"

if [ ! -x "$VENV/bin/python" ] || [ "$(cat "$STATE" 2>/dev/null || true)" != "$TARGET" ]; then
    if [ -d "$VENV" ] && [ -n "$(ls -A "$VENV" 2>/dev/null || true)" ]; then
        echo "[nlp] 解释器、Unicode 数据版本或依赖有变化，重建虚拟环境..."
        find "$VENV" -mindepth 1 -maxdepth 1 -exec rm -rf {} +
    fi
    echo "[nlp] 安装 Python 依赖（首次启动需要几分钟，装完后不再需要联网）..."
    mark 02-deps-install-started
    python -m venv "$VENV"
    if "$VENV/bin/python" -m pip install --no-cache-dir --disable-pip-version-check \
            --retries 5 --timeout 60 -i "$PIP_MIRROR" -r "$APP_DIR/requirements.txt" >"$LOG" 2>&1; then
        :
    elif "$VENV/bin/python" -m pip install --no-cache-dir --disable-pip-version-check \
            --retries 5 --timeout 60 -r "$APP_DIR/requirements.txt" >>"$LOG" 2>&1; then
        :
    else
        echo "[nlp] 依赖安装失败，日志末尾："
        tail -n 40 "$LOG" || true
        mark 03-deps-install-failed
        exit 1
    fi
    printf '%s\n' "$TARGET" > "$STATE"
    mark 03-deps-ready
    echo "[nlp] 依赖安装完成。"
else
    mark 03-deps-cached
    echo "[nlp] 依赖已就绪。"
fi

if [ "$#" -eq 0 ]; then
    set -- "$VENV/bin/python" -m streamlit run "$APP_DIR/app.py" \
        --server.address=0.0.0.0 \
        --server.port="$PORT" \
        --server.headless=true \
        --server.fileWatcherType=none \
        --server.maxUploadSize=30 \
        --browser.gatherUsageStats=false
fi

# 后台探测自身健康状态，成功后留下 05-healthy 标记。
(
    i=0
    while [ "$i" -lt 300 ]; do
        if "$VENV/bin/python" -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:$PORT/_stcore/health', timeout=5).read().strip()==b'ok' else 1)" >/dev/null 2>&1; then
            mark 05-healthy
            break
        fi
        sleep 2
        i=$((i + 1))
    done
) &

mark 04-streamlit-starting

if [ -n "${PUID:-}" ] && [ "$PUID" != "0" ] && [ "$(id -u)" = "0" ]; then
    if command -v setpriv >/dev/null 2>&1; then
        echo "[nlp] 以 UID=$PUID GID=${PGID:-$PUID} 运行。"
        exec setpriv --reuid="$PUID" --regid="${PGID:-$PUID}" --clear-groups "$@"
    fi
    echo "[nlp] 未找到 setpriv，改用 root 运行。"
fi

exec "$@"
