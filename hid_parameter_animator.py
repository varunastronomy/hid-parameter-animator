#!/usr/bin/env python3
"""Create an adjacent HID and a user-selected X-versus-Y GIF."""

import argparse
import csv
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.time import Time
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.lines import Line2D
from scipy.stats import spearmanr


DEFAULT_CSV = Path("Final.csv")
DEFAULT_FLC_LIST = Path("flc.txt")
DEFAULT_OUTPUT_DIRECTORY = Path("Adjacent_HID_parameter_GIFs")

QUARTER_MONTHS = {
    1: "Jan-Mar",
    2: "Apr-Jun",
    3: "Jul-Sep",
    4: "Oct-Dec",
}

QUARTER_COLORS = {
    "Jan-Mar 2021": "#0072B2",
    "Apr-Jun 2021": "#E69F00",
    "Jul-Sep 2021": "#009E73",
    "Oct-Dec 2021": "#D55E00",
    "Jan-Mar 2022": "#6F2DA8",
    "Apr-Jun 2022": "#8B4513",
    "Jul-Sep 2022": "#CC0077",
    "Oct-Dec 2022": "#FFD700",
}

LABELS = {
    "HID_intensity": "Intensity",
    "HID_hardness": "Hardness ratio",
    "MJD": "MJD",
    "Total_flux_e-10": r"Total flux ($10^{-10}$ erg cm$^{-2}$ s$^{-1}$)",
    "nH": r"$N_{\rm H}$",
    "kT": r"$kT_{\rm BB}$ (keV)",
    "bbodyrad_norm": r"$N_{\rm BB}$",
    "Tin": r"$kT_{\rm in}$ (keV)",
    "diskbb_norm": r"$N_{\rm disk}$",
    "bbodyrad_flux_e-10": r"$F_{\rm BB}$ ($10^{-10}$ erg cm$^{-2}$ s$^{-1}$)",
    "diskbb_flux_e-10": r"$F_{\rm disc}$ ($10^{-10}$ erg cm$^{-2}$ s$^{-1}$)",
    "bbodyrad_R_app_km": r"$R_{\rm BB,app}$ (km)",
    "bbodyrad_R_colorcorr_km": r"$R_{\rm BB,corr}$ (km)",
    "diskbb_Rin_app_km": r"$R_{\rm in,app}$ (km)",
    "diskbb_Rin_corr_km": r"$R_{\rm in,corr}$ (km)",
    "Reduced_chi_square": r"Reduced $\chi^2$",
    "bbodyrad_flux_e-10_over_Total_flux_e-10": r"$F_{\rm BB}/F_{\rm total}$",
    "diskbb_flux_e-10_over_Total_flux_e-10": r"$F_{\rm disc}/F_{\rm total}$",
    "kT_over_Tin": r"$kT_{\rm BB}/kT_{\rm in}$",
    "Fbb_over_Fdisk": r"$F_{\rm BB}/F_{\rm disc}$",
    "Rbb_app_over_Rin_app": r"$R_{\rm BB,app}/R_{\rm in,app}$",
    "Rbb_corr_over_Rin_corr": r"$R_{\rm BB,corr}/R_{\rm in,corr}$",
}

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "STIXGeneral"],
        "mathtext.fontset": "stix",
        "font.size": 15,
        "axes.labelsize": 16,
        "axes.titlesize": 16,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
    }
)


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--flc-list", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    parser.add_argument("--x", dest="x_quantity")
    parser.add_argument("--y", dest="y_quantity")
    parser.add_argument("--fps", type=int, default=2)
    parser.add_argument("--dpi", type=int, default=90)
    return parser.parse_args()


def resolve_from_cwd(path):
    path = Path(path).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    return path.resolve()


def choose_file(label, command_line_path, default):
    if command_line_path is None:
        entered = input(f"{label} [{default}]: ").strip()
        command_line_path = Path(entered) if entered else default
    path = resolve_from_cwd(command_line_path)
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")
    return path


def print_quantity_choices(quantity_names):
    print("\nAvailable quantities for either axis:")
    for number, name in enumerate(quantity_names, start=1):
        print(f"  {number:2d}. {name}")


def choose_quantity(axis_name, command_line_choice, quantity_names, default):
    choice = command_line_choice
    if choice is None:
        choice = input(
            f"Choose the {axis_name}-axis by number or name [{default}]: "
        ).strip()
    if not choice:
        return default

    if str(choice).isdigit():
        index = int(choice) - 1
        if 0 <= index < len(quantity_names):
            return quantity_names[index]
        raise ValueError(
            f"{axis_name}-axis number must be between 1 and {len(quantity_names)}."
        )

    exact_matches = [
        name for name in quantity_names if name.lower() == str(choice).lower()
    ]
    if exact_matches:
        return exact_matches[0]
    raise ValueError(f"Unknown {axis_name}-axis quantity: {choice}")


def parse_cell(cell):
    parts = [part.strip() for part in str(cell).split(",")]
    value = float(parts[0])
    lower_error = abs(float(parts[1])) if len(parts) > 1 else 0.0
    upper_error = abs(float(parts[2])) if len(parts) > 2 else lower_error
    return value, lower_error, upper_error


def load_spectral_table(csv_path):
    with csv_path.open(newline="") as handle:
        rows = list(csv.reader(handle, delimiter="\t"))
    if len(rows) < 2 or len(rows[0]) < 2:
        raise ValueError(f"No spectral data found in {csv_path}")

    obsids = np.array(rows[0][1:])
    values = {}
    for row in rows[1:]:
        values[row[0]] = np.array(
            [parse_cell(cell)[0] for cell in row[1:]], dtype=float
        )

    missing = {"Total_flux_e-10", "MJD"} - values.keys()
    if missing:
        raise KeyError("Missing CSV rows: " + ", ".join(sorted(missing)))
    return obsids, values


def add_ratio(values, numerator, denominator, output_name):
    ratio = np.full_like(values[numerator], np.nan, dtype=float)
    valid = (
        np.isfinite(values[numerator])
        & np.isfinite(values[denominator])
        & (values[denominator] != 0)
    )
    ratio[valid] = values[numerator][valid] / values[denominator][valid]
    values[output_name] = ratio


def add_derived_ratios(values):
    add_ratio(values, "kT", "Tin", "kT_over_Tin")
    add_ratio(
        values,
        "bbodyrad_flux_e-10",
        "diskbb_flux_e-10",
        "Fbb_over_Fdisk",
    )
    add_ratio(
        values,
        "bbodyrad_R_app_km",
        "diskbb_Rin_app_km",
        "Rbb_app_over_Rin_app",
    )
    add_ratio(
        values,
        "bbodyrad_R_colorcorr_km",
        "diskbb_Rin_corr_km",
        "Rbb_corr_over_Rin_corr",
    )


def get_obsid(filename):
    match = re.search(r"\d{10}", str(filename))
    if match is None:
        raise ValueError(f"No ten-digit ObsID found in {filename}")
    return match.group()


def weighted_mean(values, errors):
    values = np.asarray(values, dtype=float)
    errors = np.asarray(errors, dtype=float)
    valid = np.isfinite(values) & np.isfinite(errors) & (errors > 0)
    if np.any(valid):
        weights = errors[valid] ** -2
        return float(np.average(values[valid], weights=weights))
    return float(np.nanmean(values))


def resolve_hid_path(text, list_file):
    path = Path(text).expanduser()
    if path.is_absolute():
        return path
    relative_to_list = list_file.parent / path
    if relative_to_list.exists():
        return relative_to_list.resolve()
    return resolve_from_cwd(path)


def load_observation_averaged_hid(list_file, allowed_obsids):
    allowed_obsids = set(allowed_obsids)
    hid = {}
    for line in list_file.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        filename = resolve_hid_path(line, list_file)
        obsid = get_obsid(filename)
        if obsid not in allowed_obsids:
            continue
        if obsid in hid:
            raise RuntimeError(f"Duplicate HID file for ObsID {obsid}")
        with fits.open(filename) as hdul:
            data = hdul["RATE"].data
            intensity = weighted_mean(data["SUM12"], data["SUM_E"])
            hardness = weighted_mean(data["RATIO_12"], data["RAT_E12"])
        hid[obsid] = {"intensity": intensity, "hardness": hardness}

    missing = sorted(allowed_obsids - set(hid))
    if missing:
        raise RuntimeError(
            "No HID FITS file found for these CSV ObsIDs:\n" + "\n".join(missing)
        )
    return hid


def safe_limits(values, padding_fraction=0.08):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    lower = float(np.min(values))
    upper = float(np.max(values))
    width = upper - lower
    if width == 0:
        width = abs(lower) if lower != 0 else 1.0
    padding = padding_fraction * width
    return lower - padding, upper + padding


def filename_slug(text):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_")


def quarter_label_from_mjd(mjd):
    observation_date = Time(float(mjd), format="mjd").to_datetime()
    quarter = (observation_date.month - 1) // 3 + 1
    label = f"{QUARTER_MONTHS[quarter]} {observation_date.year}"
    if label not in QUARTER_COLORS:
        raise ValueError(
            f"MJD {mjd} is outside the configured 2021--2022 quarters."
        )
    return label


def ordered_quarters(plotted_quarters):
    present = set(plotted_quarters)
    labels = []
    for quarter in range(1, 5):
        for year in (2021, 2022):
            label = f"{QUARTER_MONTHS[quarter]} {year}"
            if label in present:
                labels.append(label)
    return labels


def quarter_legend_handles(plotted_quarters):
    handles = []
    for label in ordered_quarters(plotted_quarters):
        marker = "o" if label.endswith("2021") else "^"
        handles.append(
            Line2D(
                [],
                [],
                marker=marker,
                linestyle="none",
                markerfacecolor=QUARTER_COLORS[label],
                markeredgecolor="black",
                color=QUARTER_COLORS[label],
                markersize=7,
                label=label,
            )
        )
    return handles


def update_scatter(artist, x, y, colours, selection):
    if np.any(selection):
        artist.set_offsets(np.column_stack((x[selection], y[selection])))
        artist.set_facecolors(colours[selection])
    else:
        artist.set_offsets(np.empty((0, 2), dtype=float))
        artist.set_facecolors([])


def make_adjacent_gif(
    x_name,
    y_name,
    obsids,
    years,
    quarters,
    mjd,
    intensity,
    hardness,
    x_values,
    y_values,
    colours,
    output_file,
    fps,
    dpi,
):
    valid = (
        np.isfinite(intensity)
        & np.isfinite(hardness)
        & np.isfinite(x_values)
        & np.isfinite(y_values)
    )
    if not np.all(valid):
        raise ValueError(
            f"Non-finite {x_name} or {y_name} values for: "
            + ", ".join(obsids[~valid])
        )

    rho, pvalue = spearmanr(x_values, y_values)
    fig, (hid_ax, parameter_ax) = plt.subplots(
        1, 2, figsize=(13.5, 6.4)
    )
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.11, top=0.72, wspace=0.27)

    hid_ax.set_xlabel("Intensity")
    hid_ax.set_ylabel("Hardness ratio")
    hid_ax.set_title("Observation-averaged HID")
    hid_ax.set_xlim(*safe_limits(intensity))
    hid_ax.set_ylim(*safe_limits(hardness))

    parameter_ax.set_xlabel(LABELS.get(x_name, x_name))
    parameter_ax.set_ylabel(LABELS.get(y_name, y_name))
    parameter_ax.set_title(rf"$\rho={rho:.3f}$, $p={pvalue:.3g}$")
    parameter_ax.set_xlim(*safe_limits(x_values))
    parameter_ax.set_ylim(*safe_limits(y_values))

    for axis in (hid_ax, parameter_ax):
        axis.minorticks_on()
        axis.tick_params(which="both", direction="in", top=True, right=True)

    hid_artists = {}
    parameter_artists = {}
    for year, marker in ((2021, "o"), (2022, "^")):
        hid_artists[year] = hid_ax.scatter(
            [], [], marker=marker, s=58, edgecolors="black", linewidths=0.45
        )
        parameter_artists[year] = parameter_ax.scatter(
            [], [], marker=marker, s=58, edgecolors="black", linewidths=0.45
        )

    hid_current = hid_ax.scatter(
        [], [], s=145, facecolors="none", edgecolors="black", linewidths=2
    )
    parameter_current = parameter_ax.scatter(
        [], [], s=145, facecolors="none", edgecolors="black", linewidths=2
    )
    status = fig.suptitle("", y=0.98)
    fig.legend(
        handles=quarter_legend_handles(quarters),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.90),
        ncol=4,
        frameon=False,
        fontsize=11,
        handletextpad=0.4,
        columnspacing=1.0,
    )

    def update(frame_index):
        end = frame_index + 1
        for year in (2021, 2022):
            selection = years[:end] == year
            update_scatter(
                hid_artists[year],
                intensity[:end],
                hardness[:end],
                colours[:end],
                selection,
            )
            update_scatter(
                parameter_artists[year],
                x_values[:end],
                y_values[:end],
                colours[:end],
                selection,
            )
        hid_current.set_offsets([[intensity[frame_index], hardness[frame_index]]])
        parameter_current.set_offsets(
            [[x_values[frame_index], y_values[frame_index]]]
        )
        status.set_text(
            f"{end}/{len(obsids)} observations | ObsID {obsids[frame_index]} "
            f"| MJD {mjd[frame_index]:.2f} | {quarters[frame_index]}"
        )
        return (
            *hid_artists.values(),
            *parameter_artists.values(),
            hid_current,
            parameter_current,
            status,
        )

    animation = FuncAnimation(
        fig,
        update,
        frames=range(len(obsids)),
        interval=1000 / fps,
        blit=False,
        repeat=True,
    )
    animation.save(output_file, writer=PillowWriter(fps=fps), dpi=dpi)
    plt.close(fig)


def main():
    args = arguments()
    if args.fps < 1:
        raise ValueError("--fps must be at least 1")

    csv_path = choose_file("CSV file", args.csv, DEFAULT_CSV)
    flc_list = choose_file("HID FLC list", args.flc_list, DEFAULT_FLC_LIST)
    output_directory = resolve_from_cwd(args.output_dir)
    output_directory.mkdir(parents=True, exist_ok=True)

    obsids, values = load_spectral_table(csv_path)
    add_derived_ratios(values)
    hid = load_observation_averaged_hid(flc_list, obsids)

    order = np.argsort(values["MJD"])
    obsids = obsids[order]
    mjd = values["MJD"][order]
    intensity = np.array([hid[obsid]["intensity"] for obsid in obsids])
    hardness = np.array([hid[obsid]["hardness"] for obsid in obsids])
    quarters = np.array([quarter_label_from_mjd(value) for value in mjd])
    years = np.array([int(label[-4:]) for label in quarters], dtype=int)
    colours = np.array([QUARTER_COLORS[label] for label in quarters])

    quantities = {
        "HID_intensity": intensity,
        "HID_hardness": hardness,
    }
    for name, parameter_values in values.items():
        quantities[name] = parameter_values[order]
    quantity_names = list(quantities)

    print(f"Matched observations: {len(obsids)}")
    print_quantity_choices(quantity_names)
    x_name = choose_quantity(
        "X", args.x_quantity, quantity_names, "Total_flux_e-10"
    )
    y_name = choose_quantity(
        "Y",
        args.y_quantity,
        quantity_names,
        "bbodyrad_flux_e-10_over_Total_flux_e-10",
    )
    if x_name == y_name:
        raise ValueError("The X-axis and Y-axis quantities must be different.")

    output_file = output_directory / (
        f"HID_and_{filename_slug(y_name)}_vs_{filename_slug(x_name)}.gif"
    )
    print(f"\nX-axis: {x_name}")
    print(f"Y-axis: {y_name}")
    print("Quarter colours:")
    for label in ordered_quarters(quarters):
        print(f"  {label}: {QUARTER_COLORS[label]}")
    make_adjacent_gif(
        x_name,
        y_name,
        obsids,
        years,
        quarters,
        mjd,
        intensity,
        hardness,
        quantities[x_name],
        quantities[y_name],
        colours,
        output_file,
        args.fps,
        args.dpi,
    )
    print(f"Created: {output_file}")


if __name__ == "__main__":
    main()
