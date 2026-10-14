# Always-on stream image: generator, assembly, disclaimer TTS (espeak-ng) and ffmpeg.
# Stages 2-3 (WhisperX, pyannote) need torch and usually a GPU; build with --build-arg WITH_ALIGN=1 for those.
FROM python:3.11-slim

ARG WITH_ALIGN=0
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg espeak-ng \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml ./
COPY mvs ./mvs
RUN pip install --no-cache-dir . && if [ "$WITH_ALIGN" = "1" ]; then pip install --no-cache-dir ".[align]"; fi
COPY speakers ./speakers
COPY config ./config

# Mount the speaker's built artifacts (clips, chain.json, ...) at /app/data.
VOLUME ["/app/data"]
ENV PYTHONUNBUFFERED=1
EXPOSE 8090
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s CMD python -c "import urllib.request,sys; urllib.request.urlopen('http://127.0.0.1:8090/health'); sys.exit(0)"
ENTRYPOINT ["mvs"]
CMD ["--help"]
