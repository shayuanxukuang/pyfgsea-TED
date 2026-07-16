FROM python:3.12-slim AS builder

WORKDIR /build
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential cargo rustc \
    && rm -rf /var/lib/apt/lists/* \
    && python -m pip install --no-cache-dir "maturin>=1.8"

COPY pyproject.toml Cargo.toml Cargo.lock LICENSE ./
COPY src ./src
COPY pyfgsea ./pyfgsea
RUN maturin build --release --out /wheels

FROM python:3.12-slim

WORKDIR /workspace
RUN python -m pip install --no-cache-dir \
    numpy pandas scipy scikit-learn statsmodels matplotlib seaborn \
    pyyaml click jsonschema anndata requests openpyxl pyarrow zarr tabulate
COPY --from=builder /wheels /wheels
RUN python -m pip install --no-cache-dir --no-deps /wheels/*.whl

COPY scripts ./scripts
COPY schemas ./schemas
COPY config ./config
COPY reproducibility ./reproducibility
COPY tests ./tests

WORKDIR /run

CMD ["bash", "-lc", "python /workspace/scripts/run_ted_validation_demo.py --outdir /tmp/ted_demo && ted validate /tmp/ted_demo/demo_events_v2.tsv --kind event --report /tmp/ted_demo/event_validation.tsv"]
