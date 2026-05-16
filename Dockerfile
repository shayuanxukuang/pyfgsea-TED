FROM mambaorg/micromamba:1.5.10

WORKDIR /workspace

COPY environment.yml /tmp/environment.yml
RUN micromamba install -y -n base -f /tmp/environment.yml && micromamba clean -a -y

COPY . /workspace

ENV PATH=/opt/conda/bin:$PATH
SHELL ["/bin/bash", "-lc"]

CMD ["bash", "-lc", "python reproducibility/run_minimal_demo.py && python scripts/run_full_benchmark_suite.py --quick && python reproduce_all_main_tables.py && python reproduce_all_main_figures.py"]
