# syntax=docker/dockerfile:1.7
#
# Future Vision Engine — multi-stage image (linux/amd64 and linux/arm64).
#
#   docker build -t future-vision-engine .                       # with DINOv2 model (hybrid, best accuracy)
#   docker build -t future-vision-engine --build-arg WITH_MODEL=0 .   # classical only, smaller image
#
#   docker run --rm -v "$PWD:/work" future-vision-engine --video /work/video.mp4 --output-dir /work/inspection_results
#
# Stages: deps -> build (compile + unit/e2e tests) -> model (offline DINOv2 export) -> runtime.

ARG UBUNTU_VERSION=24.04
ARG ORT_VERSION=1.20.1

# ── deps: compiler, OpenCV, ONNX Runtime ─────────────────────────────────────
FROM ubuntu:${UBUNTU_VERSION} AS deps
ARG ORT_VERSION
ARG TARGETARCH
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake pkg-config ca-certificates curl \
        libopencv-dev \
    && rm -rf /var/lib/apt/lists/*
RUN set -eux; \
    case "${TARGETARCH:-amd64}" in \
      amd64) ORT_ARCH=x64 ;; \
      arm64) ORT_ARCH=aarch64 ;; \
      *) echo "unsupported arch ${TARGETARCH}"; exit 1 ;; \
    esac; \
    curl -fsSL "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-linux-${ORT_ARCH}-${ORT_VERSION}.tgz" \
      | tar -xz -C /opt; \
    mv /opt/onnxruntime-linux-${ORT_ARCH}-${ORT_VERSION} /opt/onnxruntime

# ── build: compile and run the test suite ───────────────────────────────────
FROM deps AS build
WORKDIR /src
COPY CMakeLists.txt ./
COPY include include
COPY src src
COPY tests tests
COPY tools tools
COPY configs configs
# FVE_NATIVE=OFF: portable binaries (no -mcpu=native tied to the build host).
RUN cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DFVE_NATIVE=OFF \
        -DORT_INCLUDE=/opt/onnxruntime/include -DORT_LIB=/opt/onnxruntime/lib/libonnxruntime.so \
        | tee /tmp/configure.log \
    && grep -q "ONNX Runtime (DINOv2 backend): ON" /tmp/configure.log \
    && cmake --build build --parallel "$(nproc)"
# Tests are a build gate: a failing check fails the image build.
RUN LD_LIBRARY_PATH=/opt/onnxruntime/lib ctest --test-dir build --output-on-failure

# ── model: offline DINOv2 export (Python only lives in this stage) ──────────
FROM ubuntu:${UBUNTU_VERSION} AS model
ARG WITH_MODEL=1
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-venv ca-certificates \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /m
COPY tools/prepare_model.py .
RUN mkdir -p /m/models && if [ "$WITH_MODEL" = "1" ]; then \
        python3 -m venv /venv && /venv/bin/pip install --no-cache-dir onnx \
        && /venv/bin/python prepare_model.py --variant reg-small --layers 5 8 --out /m/models/dinov2_ml.onnx \
        && rm -f /m/models/dinov2-*.onnx; \
    fi

# ── runtime: binaries + shared libs + model, no compiler, no Python ─────────
FROM ubuntu:${UBUNTU_VERSION} AS runtime
ENV DEBIAN_FRONTEND=noninteractive
# Runtime OpenCV modules only (pulls the FFmpeg codecs used by videoio).
RUN apt-get update && apt-get install -y --no-install-recommends \
        libopencv-core406t64 libopencv-imgproc406t64 libopencv-imgcodecs406t64 \
        libopencv-videoio406t64 libopencv-features2d406t64 libopencv-calib3d406t64 \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY --from=deps /opt/onnxruntime/lib/ /opt/onnxruntime/lib/
RUN echo /opt/onnxruntime/lib > /etc/ld.so.conf.d/onnxruntime.conf && ldconfig
COPY --from=build /src/build/fve_inspect /src/build/fve_synth /opt/fve/bin/
COPY --from=build /src/configs /opt/fve/configs
COPY --from=model /m/models /opt/fve/models
# fve_inspect auto-loads <exe>/../models/dinov2_ml.onnx when present.
RUN useradd --create-home --uid 10001 fve && mkdir -p /work && chown fve /work
USER fve
WORKDIR /work
ENV PATH=/opt/fve/bin:$PATH
ENTRYPOINT ["fve_inspect"]
CMD ["--help"]
