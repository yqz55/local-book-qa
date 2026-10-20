# 本地/飞牛 NAS 通用镜像：先把依赖装进 /opt/venv，容器启动时不再联网安装。
# Python 版本需与生成索引/复算记录的版本一致：3.13 → Unicode 15.1.0，
# 该版本写入记录（identity.unicode_version），不一致会无法复算旧记录。
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    NLP_DATA_DIR=/app/.data

WORKDIR /app

COPY requirements.txt ./
RUN python -m venv /opt/venv \
    && /opt/venv/bin/python -m pip install --no-cache-dir --disable-pip-version-check \
        --retries 5 --timeout 60 -r requirements.txt \
    && printf '%s|%s|%s\n' \
        "$(python -c 'import sys; print("%d.%d" % sys.version_info[:2])')" \
        "$(python -c 'import unicodedata; print(unicodedata.unidata_version)')" \
        "$(sha256sum requirements.txt | cut -d' ' -f1)" > /opt/venv/.nlp-env

COPY . /app

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=4).read().strip()==b'ok' else 1)"

ENTRYPOINT ["/bin/sh", "/app/deploy/fnos/entrypoint.sh"]
