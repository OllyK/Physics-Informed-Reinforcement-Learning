import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def scan_and_plot_best(root_dir, show=True, save_plot=None):
    """
    Scan recursively for .npy files (old raw array or new dict format),
    select best by primary efficiency (margin if multi-RI, else eff),
    plot structure, print efficiencies, return (structure, info).
    """
    root_dir = Path(root_dir)
    if not root_dir.exists():
        raise FileNotFoundError(f"Directory not found: {root_dir}")

    best = None  # (primary_eff, structure, info, filepath)
    for fp in root_dir.rglob('*.npy'):
        try:
            data = np.load(fp, allow_pickle=True)
        except Exception:
            continue

        info = {}
        structure = None
        primary_eff = None

        # New format dict?
        if data.shape == () and isinstance(data.item(), dict):
            d = data.item()
            structure = np.array(d.get('structure'))
            eff = float(d.get('eff', 0.0))
            eff_on = d.get('eff_on')
            eff_off = d.get('eff_off')
            if eff_on is not None and eff_off is not None:
                info = {'eff_on': float(eff_on), 'eff_off': float(eff_off), 'margin': eff}
                primary_eff = eff
            else:
                info = {'eff': eff}
                primary_eff = eff
        else:
            # Old format: raw array; derive efficiency from filename
            structure = np.array(data)
            name = fp.stem
            parts = name.split('_')
            if parts[0].startswith('on') and any(p.startswith('m') for p in parts):
                def restore(tag):
                    for i, ch in enumerate(tag):
                        if ch.isdigit():
                            num = tag[i:]
                            break
                    else:
                        return 0.0
                    if '-' in num:
                        num = num.replace('-', '.', 1)
                    try:
                        return float(num) / 100.0
                    except ValueError:
                        return 0.0
                try:
                    eff_on = restore(parts[0])
                    eff_off = restore([p for p in parts if p.startswith('off')][0])
                    margin = restore([p for p in parts if p.startswith('m')][0])
                except Exception:
                    continue
                info = {'eff_on': eff_on, 'eff_off': eff_off, 'margin': margin}
                primary_eff = margin
            else:
                token = parts[0]
                if '-' in token:
                    token = token.replace('-', '.', 1)
                try:
                    eff = float(token) / 100.0
                except ValueError:
                    eff = 0.0
                info = {'eff': eff}
                primary_eff = eff

        if structure is None or primary_eff is None:
            continue
        if (best is None) or (primary_eff > best[0]):
            best = (primary_eff, structure, info, fp)

    if best is None:
        raise RuntimeError(f"No valid .npy output files found under {root_dir}")

    _, structure, info, fp = best

    fig, ax = plt.subplots(figsize=(10, 2))
    ax.imshow(structure[np.newaxis, :], aspect='auto', cmap='bwr', vmin=-1, vmax=1)
    ax.set_yticks([])
    ax.set_xlabel('Cell Index')
    if 'margin' in info:
        title = f"Margin={info['margin']:.4f} | On={info['eff_on']:.4f} | Off={info['eff_off']:.4f}"
    else:
        title = f"Eff={info['eff']:.4f}"
    ax.set_title(f"{title} (File: {fp.name})")
    plt.tight_layout()
    if save_plot:
        fig.savefig(save_plot, dpi=150)
    if show:
        plt.show()
    else:
        plt.close(fig)

    print("Best file:", fp)
    print("Structure array:\n", structure)
    print("Efficiencies:")
    for k, v in info.items():
        print(f"  {k}: {v:.6f}")

    return structure, info


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Scan metasurface .npy outputs (old/new formats) to find and plot best structure."
    )
    parser.add_argument("directory", help="Root directory to scan.")
    parser.add_argument("--save-plot", dest="save_plot", default=None,
                        help="Path to save plot (e.g. best.png).")
    parser.add_argument("--no-show", action="store_true",
                        help="Do not display the plot (headless use).")
    args = parser.parse_args()

    scan_and_plot_best(
        args.directory,
        show=not args.no_show,
        save_plot=args.save_plot
    )
