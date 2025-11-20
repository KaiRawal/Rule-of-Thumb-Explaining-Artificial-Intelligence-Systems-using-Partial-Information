import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import argparse

parser = argparse.ArgumentParser(description="Plot runtimes with scaling and fit options.")
parser.add_argument("--xscale", choices=["linear", "log"], default="log",
                    help="Scale for x-axis (default: log)")
parser.add_argument("--y1scale", choices=["linear", "log"], default="log",
                    help="Scale for primary y-axis (default: log)")
parser.add_argument("--y2scale", choices=["linear", "log"], default="linear",
                    help="Scale for secondary y-axis (default: linear)")
parser.add_argument("--fit1", choices=["linear", "lin-log", "log-lin", "log-log"], default="log-log",
                    help="Fit type for primary axis (default: linear)")
parser.add_argument("--fit2", choices=["linear", "lin-log", "log-lin", "log-log"], default="lin-log",
                    help="Fit type for secondary axis (default: linear)")

args = parser.parse_args()

xscale = args.xscale
y1scale = args.y1scale
y2scale = args.y2scale
fit1 = args.fit1
fit2 = args.fit2


# --- Helper for fits ---
def fit_and_plot(ax, x, y, fit_type, color, label):
    """Fit according to fit_type and plot a dashed line."""
    xfit = np.linspace(min(min(x), 65), max(x), 200)

    if fit_type == "linear":  
        X = x.reshape(-1,1); Y = y
        model = LinearRegression().fit(X, Y)
        yfit = model.predict(xfit.reshape(-1,1))

    elif fit_type == "lin-log":  # y = a log x + b
        X = np.log(x).reshape(-1,1); Y = y
        model = LinearRegression().fit(X, Y)
        yfit = model.predict(np.log(xfit).reshape(-1,1))

    elif fit_type == "log-lin":  # log y = ax + b
        X = x.reshape(-1,1); Y = np.log(y)
        model = LinearRegression().fit(X, Y)
        yfit = np.exp(model.predict(xfit.reshape(-1,1)))

    elif fit_type == "log-log":  # log y = a log x + b
        X = np.log(x).reshape(-1,1); Y = np.log(y)
        model = LinearRegression().fit(X, Y)
        yfit = np.exp(model.predict(np.log(xfit).reshape(-1,1)))

    else:
        raise ValueError(f"Unknown fit type: {fit_type}")

    ax.plot(xfit, yfit, ":", color=color, alpha=0.85, zorder=1, linewidth=3)

# =====================
# Load + preprocess data
# =====================
df = pd.read_csv('./RunTimes.csv')

mask = df["INFO"].str.contains("LR|ANN", regex=True, na=False)
df.loc[mask, "total_lime"] = (df.loc[mask, "total_lime"] - df.loc[mask, "init_lime"]) * 5 + df.loc[mask, "init_lime"]

df = df.sort_values(by='datapoints')

df['rot_total_a'] = (df['total_rot_s'] + df['total_rot_l']) / 2
df['speedup_s'] = df['total_shap'] / df['total_rot_s']
df['speedup_l'] = df['total_lime'] / df['total_rot_l']

num_explanations = df['datapoints'].values
rot_total = df['rot_total_a'].values
shap_total = df['total_shap'].values
lime_total = df['total_lime'].values
speedup_s = df['speedup_s'].values
speedup_l = df['speedup_l'].values

# =====================
# Plotting
# =====================
fig, ax1 = plt.subplots(figsize=(8,6))

ax1.set_xscale(xscale)
ax1.set_yscale(y1scale)

# --- Primary axis (runtimes) ---
ax1.scatter(num_explanations, rot_total, color="C0", label="RoT", marker='.', alpha=0.95)
ax1.scatter(num_explanations, shap_total, color="C1", label="SHAP", marker=".", alpha=0.95)
ax1.scatter(num_explanations, lime_total, color="C2", label="LIME", marker=".", alpha=0.95)

# best-fits
fit_and_plot(ax1, num_explanations, rot_total, fit1, "C0", "RoT")
fit_and_plot(ax1, num_explanations, shap_total, fit1, "C1", "SHAP")
fit_and_plot(ax1, num_explanations, lime_total, fit1, "C2", "LIME")

ax1.set_xlabel(f"Number of Explanations ({xscale})", fontsize=20)
ax1.set_ylabel(f"Total Runtime ({y1scale})", fontsize=20)

# --- Secondary axis (speedup) ---
ax2 = ax1.twinx()
ax2.set_yscale(y2scale)

ax2.scatter(num_explanations, speedup_s, color="C5", alpha=0.9, marker='x', label="Speedup SHAP")
ax2.scatter(num_explanations, speedup_l, color="C6", alpha=0.9, marker='x', label="Speedup LIME")

# best-fits
fit_and_plot(ax2, num_explanations, speedup_s, fit2, "C5", "Speedup SHAP")
fit_and_plot(ax2, num_explanations, speedup_l, fit2, "C6", "Speedup LIME")

ax2.set_ylabel(f"RoT Speedup Factor ({y2scale})", fontsize=20)

fig.tight_layout()
ax1.legend(loc="upper left", fontsize=12)
ax2.legend(loc="upper right", fontsize=12)









# =====================
# Standalone plots
# =====================

# --- Ax1 only ---
fig1, ax1_only = plt.subplots(figsize=(6,6))
ax1_only.set_xscale(xscale)
ax1_only.set_yscale(y1scale)

ax1_only.scatter(num_explanations, rot_total, color="C0", label="RoT", marker='.', alpha=0.95, zorder=1, s=95)
ax1_only.scatter(num_explanations, shap_total, color="C1", label="SHAP", marker="x", alpha=0.95, zorder=1, s=105)
ax1_only.scatter(num_explanations, lime_total, color="C2", label="LIME", marker="1", alpha=0.95, zorder=1, s=130)

fit_and_plot(ax1_only, num_explanations, rot_total, fit1, "C0", "RoT")
fit_and_plot(ax1_only, num_explanations, shap_total, fit1, "C1", "SHAP")
fit_and_plot(ax1_only, num_explanations, lime_total, fit1, "C2", "LIME")

ax1_only.set_xlabel(f"Number of Explanations ({xscale})", fontsize=24)
ax1_only.set_ylabel(f"Total Runtime ({y1scale})", fontsize=24)
ax1_only.legend(fontsize=19, borderaxespad=0.1, borderpad=0.2)
ax1_only.tick_params(axis='both', labelsize=19)

# _xlims = ax1_only.get_xlim()
# ax1_only.set_xlim((50, _xlims[1]))

plt.tight_layout()
plt.savefig('runtimes_ax1.pdf', dpi=300)
plt.close(fig1)

# --- Ax2 only ---
fig2, ax2_only = plt.subplots(figsize=(6,6))
ax2_only.set_xscale(xscale)
ax2_only.set_yscale(y2scale)

ax2_only.scatter(num_explanations, speedup_s, color="darkgreen", alpha=0.95, marker='x', label="SHAP", s=105)
ax2_only.scatter(num_explanations, speedup_l, color="fuchsia", alpha=0.95, marker='1', label="LIME", s=130)

fit_and_plot(ax2_only, num_explanations, speedup_s, fit2, "darkgreen", "SHAP")
fit_and_plot(ax2_only, num_explanations, speedup_l, fit2, "fuchsia", "LIME")

ax2_only.set_xlabel(f"Number of Explanations ({xscale})", fontsize=24)
ax2_only.set_ylabel(f"RoT Speedup Factor", fontsize=24)
ax2_only.legend(fontsize=19, title="Speedup Relative To", title_fontsize=19, borderaxespad=0.1, borderpad=0.2)
ax2_only.tick_params(axis='both', labelsize=19)
# _xlims = ax2_only.get_xlim()
# ax2_only.set_xlim((1, _xlims[1]))

plt.tight_layout()
plt.savefig('./runtimes_ax2.pdf', dpi=300)
plt.close(fig2)


