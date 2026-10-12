#!/usr/bin/env bash
# Run all experiments of the paper in parallel, one process per experiment and sweep value
# (23 in total), and combine the epsilon, beta, and n_y sweeps into one experiment each at the end.
#
# Usage, from any directory, with the virtual environment activated:
#
#     bash scripts/run_paper_experiments.sh
#
# Experiment names (results/<name>_*, figures/fig_<name>_*, tikz_figures/*<name>*):
#     mainHc_      Sec. 4, high-confidence guarantee (epsilon = 0.05, beta = 1e-6)
#     appHc_       App. E, high-confidence guarantee: beta sweep, epsilon sweeps, n_y sweep
# The sweeps are combined into appHc_Vehicle30_eps5_betasweep,
# appHc_Vehicle30_epssweep_beta1e-06, appHc_CASP_epssweep_beta1e-06, and
# appHc_SCM20Ddimsweep_eps5_beta1e-06, appExpect_Vehicle30_epssweep, and
# appExpect_CASP_epssweep (one panel per value).
#     appExpect_   App. F, expected guarantee: vehicle T = 2, 30, 50 and
#                  epsilon sweeps for vehicle T = 30 and CASP
# The point epsilon = 0.05, beta = 1e-6 of the vehicle sweeps (T = 30) is the Sec. 4
# experiment mainHc_Vehicle30_eps5_beta1e-06; it is run once and shared by both sweeps.
# Likewise, the point epsilon = 0.05 of the expected-guarantee vehicle sweep is
# appExpect_Vehicle30_eps5.
#
# Per-experiment logs go to results/*_log.txt, start-up errors and tracebacks to
# results/run_*.out, the output of the combination step to results/run_combine.out, and
# the code version and machine to results/run_info.txt.  The script refuses to start if
# results/ already contains output, so that results of different code versions are not
# mixed; remove or rename results/ first.
#
# Environment variables:
#     PYTHON=...   Python interpreter (default: python)
#     DRY_RUN=1    only print the commands
#     N_RUNS=...   calibration runs per target (default: 500, as in the paper)

set -u
cd "$(dirname "$0")/.." || exit 1

PYTHON=${PYTHON:-python}
N_RUNS=${N_RUNS:-500}
# One thread per process: the experiments run in parallel.
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
# Unbuffered output, so that results/run_*.out show the progress immediately.
export PYTHONUNBUFFERED=1

if [ -z "${DRY_RUN:-}" ] && [ -d results ] && [ -n "$(ls -A results)" ]; then
    echo "results/ is not empty; remove or rename it before starting new runs." >&2
    exit 1
fi

names=()
pids=()
run() {
    local name=$1
    shift
    local cmd=("$PYTHON" -m MCP.experiment --n-runs "$N_RUNS" "$@")
    if [ -n "${DRY_RUN:-}" ]; then
        echo "$name: ${cmd[*]}"
        return
    fi
    nohup "${cmd[@]}" > "results/run_$name.out" 2>&1 &
    names+=("$name")
    pids+=($!)
}

# Combination of the sweeps after all experiments: combine NAME EXPERIMENT... [-- LABEL...],
# with the panels in the given order and, for experiments with the same target, one LaTeX
# label per experiment.
combine() {
    local name=$1
    shift
    local paths=()
    while [ $# -gt 0 ] && [ "$1" != "--" ]; do
        paths+=("results/${1}_results.pkl")
        shift
    done
    local cmd=("$PYTHON" -m MCP.reporting.regenerate "${paths[@]}" --combine "$name")
    if [ $# -gt 0 ]; then
        shift
        cmd+=(--labels "$@")
    fi
    if [ -n "${DRY_RUN:-}" ]; then
        echo "combine: ${cmd[*]}"
    else
        "${cmd[@]}" >> results/run_combine.out 2>&1
    fi
}

if [ -z "${DRY_RUN:-}" ]; then
    mkdir -p results
    {
        echo "Started: $(date)"
        echo "Host: $(hostname)"
        echo "Commit: $(git rev-parse HEAD 2>/dev/null) ($(git rev-parse --abbrev-ref HEAD 2>/dev/null))"
        echo "Uncommitted changes:"
        git status --short 2>/dev/null
        echo "Python: $("$PYTHON" --version 2>&1)"
        if command -v lscpu > /dev/null; then
            echo "CPU: $(lscpu | grep 'Model name' | sed 's/Model name: *//')"
        else
            echo "CPU: $(sysctl -n machdep.cpu.brand_string 2>/dev/null)"
        fi
    } > results/run_info.txt
fi

# Sec. 4: high-confidence guarantee, epsilon = 0.05, beta = 1e-6
for T in 2 30 50; do
    run "mainHc_T$T" --data Vehicle --timesteps "$T" --epsilons 0.05 --betas 1e-6 \
        --name-prefix mainHc_
done

# App. E.2: beta sweep at epsilon = 0.05 (beta = 1e-6 is the Sec. 4 experiment)
for B in 1e-1 1e-3 1e-9; do
    run "appHc_beta$B" --data Vehicle --timesteps 30 --epsilons 0.05 --betas "$B" \
        --name-prefix appHc_
done

# App. E.1: epsilon sweeps at beta = 1e-6 (vehicle: epsilon = 0.05 is the Sec. 4 experiment)
for E in 0.02 0.15; do
    run "appHc_vehicle_eps$E" --data Vehicle --timesteps 30 --epsilons "$E" --betas 1e-6 \
        --name-prefix appHc_
done
for E in 0.02 0.05 0.15; do
    run "appHc_casp_eps$E" --data CASP --epsilons "$E" --betas 1e-6 --name-prefix appHc_
done

# App. E.3: varying output dimension (n_cal = 2000 is needed for beta = 1e-6)
for D in 2 4 6 8; do
    run "appHc_scm20d_dim$D" --data SCM20D --output-dims "$D" --n-cal 2000 --epsilons 0.05 \
        --betas 1e-6 --name-prefix appHc_
done

# App. F.1: expected guarantee
for T in 2 30 50; do
    run "appExpect_T$T" --data Vehicle --timesteps "$T" --epsilons 0.05 --name-prefix appExpect_
done
# App. F.2: epsilon sweeps under the expected guarantee (vehicle: epsilon = 0.05 is above)
for E in 0.02 0.15; do
    run "appExpect_vehicle_eps$E" --data Vehicle --timesteps 30 --epsilons "$E" \
        --name-prefix appExpect_
done
for E in 0.02 0.05 0.15; do
    run "appExpect_casp_eps$E" --data CASP --epsilons "$E" --name-prefix appExpect_
done

combinations() {
    combine appHc_Vehicle30_eps5_betasweep \
        appHc_Vehicle30_eps5_beta1e-01 appHc_Vehicle30_eps5_beta1e-03 \
        mainHc_Vehicle30_eps5_beta1e-06 appHc_Vehicle30_eps5_beta1e-09 || return 1
    combine appHc_Vehicle30_epssweep_beta1e-06 \
        appHc_Vehicle30_eps2_beta1e-06 mainHc_Vehicle30_eps5_beta1e-06 \
        appHc_Vehicle30_eps15_beta1e-06 || return 1
    combine appHc_CASP_epssweep_beta1e-06 \
        appHc_CASP_eps2_beta1e-06 appHc_CASP_eps5_beta1e-06 appHc_CASP_eps15_beta1e-06 || return 1
    combine appHc_SCM20Ddimsweep_eps5_beta1e-06 \
        appHc_SCM20Ddim2_eps5_beta1e-06 appHc_SCM20Ddim4_eps5_beta1e-06 \
        appHc_SCM20Ddim6_eps5_beta1e-06 appHc_SCM20Ddim8_eps5_beta1e-06 \
        -- '$n_y = 2$' '$n_y = 4$' '$n_y = 6$' '$n_y = 8$' || return 1
    combine appExpect_Vehicle30_epssweep \
        appExpect_Vehicle30_eps2 appExpect_Vehicle30_eps5 appExpect_Vehicle30_eps15 || return 1
    combine appExpect_CASP_epssweep \
        appExpect_CASP_eps2 appExpect_CASP_eps5 appExpect_CASP_eps15 || return 1
}

if [ -n "${DRY_RUN:-}" ]; then
    combinations
    exit 0
fi

echo "Started ${#pids[@]} experiments; waiting for them to finish (Ctrl-b d detaches tmux)."
failed=()
for i in "${!pids[@]}"; do
    wait "${pids[$i]}" || failed+=("${names[$i]}")
done
echo "Finished: $(date)" >> results/run_info.txt
if [ ${#failed[@]} -ne 0 ]; then
    echo "Finished; failed or stopped: ${failed[*]} (see results/run_<name>.out)." >&2
    echo "The sweeps were not combined; see scripts/run_paper_experiments.sh." >&2
    exit 1
fi
if ! combinations; then
    echo "All ${#pids[@]} experiments finished, but combining the sweeps failed" \
        "(see results/run_combine.out)." >&2
    exit 1
fi
echo "All ${#pids[@]} experiments finished successfully; sweeps combined."
