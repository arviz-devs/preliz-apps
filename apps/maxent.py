import warnings
from math import isfinite

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st
import preliz as pz
from preliz.internal.distribution_helper import init_vals

warnings.filterwarnings("ignore")

EXCLUDED_DISTS = {"Categorical", "Bernoulli"}
VALID_STATS = ["mean", "mode", "median", "var", "std", "skewness", "kurtosis"]


def _build_dist_lists():
    continuous = []
    discrete = []
    for cls in pz.distributions.all_continuous:
        if cls.__name__ not in EXCLUDED_DISTS:
            continuous.append(cls.__name__)
    for cls in pz.distributions.all_discrete:
        if cls.__name__ not in EXCLUDED_DISTS:
            discrete.append(cls.__name__)
    return sorted(continuous), sorted(discrete)


CONTINUOUS_DISTS, DISCRETE_DISTS = _build_dist_lists()

def _param_slider_bounds(dist_inst, param_name, is_int):
    idx = list(dist_inst.param_names).index(param_name)
    lo, hi = dist_inst.params_support[idx]
    dv = dist_inst.params_dict[param_name]

    s_lo = lo if isfinite(lo) else None
    s_hi = hi if isfinite(hi) else None
    step = 1 if is_int else 0.01

    if is_int:
        dv = int(dv)
        if s_lo is not None:
            s_lo = int(s_lo)
        if s_hi is not None:
            s_hi = int(s_hi)
    else:
        dv = float(dv)
        if s_lo is not None:
            s_lo = round(lo, 2)
        if s_hi is not None:
            s_hi = round(hi, 2)

    return s_lo, s_hi, dv, step


def _compute_maxent(dist_cls, fixed_params, lower, upper, mass, fixed_stat):
    dist_inst = dist_cls(**fixed_params) if fixed_params else dist_cls()
    if dist_inst.is_frozen:
        return None, "All parameters are fixed — at least one must be free for optimization."
    pz.maxent(
        dist_inst,
        lower=lower,
        upper=upper,
        mass=mass,
        fixed_stat=fixed_stat,
        plot=False,
    )
    return dist_inst, None


def _display_result(dist_inst, dist_name, mass, plot_type):
    lower = st.session_state.get("lower_bound")
    upper = st.session_state.get("upper_bound")
    fig, ax = plt.subplots(figsize=(8, 4))

    if plot_type == "CDF":
        dist_inst.plot_cdf(ax=ax, support="restricted", pointinterval=True)
        ax.set_ylabel("CDF")
    elif plot_type == "PPF":
        dist_inst.plot_ppf(ax=ax)
        ax.set_ylabel("Quantile")
    elif plot_type == "SF":
        dist_inst.plot_sf(ax=ax, support="restricted")
        ax.set_ylabel("SF")
    elif plot_type == "ISF":
        dist_inst.plot_isf(ax=ax)
        ax.set_ylabel("ISF")
    else:
        dist_inst.plot_pdf(ax=ax, support="restricted", pointinterval=True)
        ax.plot(
            [lower, upper],
            [0, 0],
            "o",
            color=ax.get_lines()[-1].get_c() if ax.get_lines() else "C0",
            alpha=0.5,
        )

    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Parameters")
    param_cols = st.columns(len(dist_inst.params_dict))
    for col, (pn, pv) in zip(param_cols, dist_inst.params_dict.items()):
        with col:
            st.metric(pn, f"{pv:.4f}")

    st.subheader("Summary")
    summary = dist_inst.summary()
    summary_cols = st.columns(len(summary._fields))
    for col, field in zip(summary_cols, summary._fields):
        with col:
            label = field
            if field == "lower":
                label = "lower_ci"
            elif field == "upper":
                label = "upper_ci"
            else:
                label = label.capitalize()
            st.metric(
                label,
                f"{getattr(summary, field):.4f}",
            )

    with st.expander("Optimization details"):
        opt = dist_inst.opt
        st.write(f"**Message**: {opt.message}")
        st.write(f"**Success**: {opt.success}")
        st.write(f"**Iterations**: {opt.nit}")
        st.write(f"**Free parameters**: {opt.x}")

        actual_mass = dist_inst.cdf(upper) - dist_inst.cdf(lower)
        st.write(f"**Computed mass**: {actual_mass:.6f}")
        st.write(f"**Requested mass**: {mass:.6f}")
        st.write(f"**Relative error**: {abs(actual_mass - mass) / mass:.6f}")


st.set_page_config(page_title="PreliZ — Maximum Entropy", layout="wide")

if "result_dist" not in st.session_state:
    st.session_state.result_dist = None
    st.session_state.result_info = {}
if "prev_dist_name" not in st.session_state:
    st.session_state.prev_dist_name = None


def run():
    """This function contains the main layout block of your maxent app."""
    if "result_dist" not in st.session_state:
        st.session_state.result_dist = None
        st.session_state.result_info = {}
    if "prev_dist_name" not in st.session_state:
        st.session_state.prev_dist_name = None

    st.title("Maxent")
    st.markdown(
        """
    Find the **maximum entropy distribution** that places a given probability mass
    between two bounds.
    
    **How it works:** Select a distribution family, set the interval and probability mass,
    and hit *Compute*. Alternatively, you can fix one or more parameters and/or a summary statistic to further
    constrain the distribution.
    """
    )

    with st.sidebar:
        st.header("Distribution")

        tab_kind = st.radio("Type", ["Continuous", "Discrete"], horizontal=True)

        if tab_kind == "Continuous":
            dist_names = CONTINUOUS_DISTS
        else:
            dist_names = DISCRETE_DISTS

        default_idx = dist_names.index("Normal") if "Normal" in dist_names else 0
        dist_name = st.selectbox("Family", dist_names, index=default_idx)

        if dist_name != st.session_state.prev_dist_name:
            st.session_state.prev_dist_name = dist_name
            st.session_state.result_dist = None
            st.session_state.result_info = {}
            for key in list(st.session_state.keys()):
                if key.startswith(("lower_", "upper_", "fix_", "val_")):
                    del st.session_state[key]

        dist_cls = getattr(pz.distributions, dist_name)
        defaults = init_vals.get(dist_name, {})
        dist_obj = dist_cls(**defaults) if defaults else dist_cls()

        st.divider()
        st.header("Fix parameters")

        param_names = dist_obj.param_names
        if isinstance(param_names, str):
            param_names = (param_names,)

        fixed_params = {}
        for pn in param_names:
            is_int = isinstance(defaults.get(pn), int)
            s_lo, s_hi, dv_adj, step = _param_slider_bounds(dist_obj, pn, is_int)

            fix = st.checkbox(f"Fix **{pn}**", key=f"fix_{pn}")
            if fix:
                val = st.number_input(
                    f"Value for {pn}",
                    min_value=s_lo,
                    max_value=s_hi,
                    value=dv_adj,
                    step=step,
                    key=f"val_{pn}",
                )
                fixed_params[pn] = val

        st.divider()
        st.header("Fix summary statistics")

        use_fixed_stat = st.checkbox("Fix a statistic")
        fixed_stat = None
        if use_fixed_stat:
            stat_name = st.selectbox("Statistic", VALID_STATS, key="stat_name")
            stat_val = st.number_input("Value", value=0.0, step=0.1, key="stat_val")
            fixed_stat = (stat_name, stat_val)

        st.divider()
        st.header("RcParams")

        ci_kind = st.selectbox(
            "CI kind", ["eti", "hdi"],
            index=["eti", "hdi"].index(pz.rcParams["stats.ci_kind"]),
            key="rc_ci_kind",
        )
        pz.rcParams["stats.ci_kind"] = ci_kind

        ci_prob = st.number_input(
            "CI prob", value=float(pz.rcParams["stats.ci_prob"]),
            min_value=0.01, max_value=1.0, step=0.01,
            key="rc_ci_prob",
        )
        pz.rcParams["stats.ci_prob"] = ci_prob

    col1, col2, col3 = st.columns(3)

    with col1:
        lower = st.number_input("Lower bound", value=None, step=0.1, key="lower_bound")
    with col2:
        upper = st.number_input("Upper bound", value=None, step=0.1, key="upper_bound")
    with col3:
        mass_val = st.number_input(
            "Probability mass", value=0.90, min_value=0.01, max_value=1.0, step=0.01,
            key="mass_val",
        )

    plot_type = st.radio("Plot type", ["PDF", "CDF", "PPF", "SF", "ISF"], horizontal=True)

    compute = st.button("Compute", type="primary", use_container_width=True)

    if compute:
        if upper <= lower:
            st.error("Upper bound must be greater than lower bound.")
            st.session_state.result_dist = None
        elif mass_val <= 0 or mass_val > 1:
            st.error("Mass must be between 0 and 1.")
            st.session_state.result_dist = None
        else:
            try:
                dist_inst, err = _compute_maxent(
                    dist_cls, fixed_params, lower, upper, mass_val, fixed_stat
                )
                if err:
                    st.error(err)
                    st.session_state.result_dist = None
                else:
                    st.session_state.result_dist = dist_inst
                    st.session_state.result_info = {
                        "dist_name": dist_name,
                        "mass": mass_val,
                    }
            except ValueError as exc:
                st.error(str(exc))
                st.session_state.result_dist = None
            except Exception as exc:
                st.error(f"Unexpected error: {exc}")
                st.session_state.result_dist = None

    if st.session_state.result_dist is not None:
        info = st.session_state.result_info
        _display_result(
            st.session_state.result_dist,
            info["dist_name"],
            info["mass"],
            plot_type,
        )
    elif not compute:
        st.info(
            "Configure the distribution and bounds on the left, then press **Compute**."
        )
