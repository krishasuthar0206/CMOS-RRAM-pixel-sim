import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator 
from matplotlib.gridspec import GridSpec
from PIL import Image
from pathlib import Path
from mpl_toolkits.axes_grid1 import make_axes_locatable


# Parameters
image_path = r'C:\Users\krisha\Desktop\RRAM_CMOS_based_image_sensor\new_image_IEDM.png'
image_size = (710, 710)

V_reset = 2.0
V_ref = 0.4  # Reference voltage for capacitance transition
T_int = 4e-7
C0 = 0.65e-15
q_e = 1.602e-19

I_max_dim = 1e-13
I_max_bright = 5e-9

noise_std = 1e-3

def load_image(image_path, size):
    if not Path(image_path).exists():
        raise FileNotFoundError(f"Image file {image_path} not found")
    img = Image.open(image_path).convert('L').resize(size)
    img_arr = np.array(img) / 255.0
    return img_arr

def dynamic_cfd(V_fd):
    A = 5.73e-15   # Scaled down to match static C0
    B = 0.237
    C = 0.176
    D = -1.02e-15
    delta_V = V_fd - V_ref
    C_fd = A * np.tanh(-B * delta_V + C) - D * delta_V
    return np.clip(C_fd, 0.1e-16, 10e-13)

def dynamic_pixel_output(I_photo, T_int):
    """Iterative solver for dynamic capacitance case (discharge model)"""
    V_fd_guess = V_reset - (I_photo * T_int) / 1e-15  # Initial guess
    for _ in range(200):  # Fixed iterations
        C_fd_dyn = dynamic_cfd(V_fd_guess)
        V_fd_guess = V_reset - (I_photo * T_int) / C_fd_dyn
    return np.clip(V_fd_guess, 0, V_reset)

def process_and_plot(image_path, image_size, I_max, scenario_label, pdf_name, exposure_type):
    img_arr = load_image(image_path, image_size)

    if exposure_type == "dim":
        ref_img = (img_arr * 0.3).clip(0, 1) * 255
    elif exposure_type == "bright":
        ref_img = (img_arr * 2).clip(0, 1) * 255
    else:
        ref_img = img_arr * 255

    I_photo = img_arr * I_max

    V_fd_static = np.clip(V_reset - (I_photo * T_int) / C0, 0, V_reset)
    # img_static = (V_fd_static * 255).astype(np.uint8)
    SNR_static = 20 * np.log10(np.maximum(V_fd_static, noise_std) / noise_std)

    V_fd_dynamic = dynamic_pixel_output(I_photo, T_int)
    # img_dynamic = (V_fd_dynamic * 255).astype(np.uint8)

    img_static = ((1 - V_fd_static / V_reset) * 255).astype(np.uint8)
    img_dynamic = ((1 - V_fd_dynamic / V_reset) * 255).astype(np.uint8)

    SNR_dynamic = 20 * np.log10(np.maximum(V_fd_dynamic, noise_std) / noise_std)

    C_fd_dynamic = dynamic_cfd(V_fd_dynamic)
    CG_dynamic = 1e6 / (C_fd_dynamic / q_e)
    CG_static = 1e6 / (C0 / q_e)
    CG_diff = CG_dynamic - CG_static
    SNR_diff = SNR_dynamic - SNR_static

    import pandas as pd
    def save_vfd_cfd_to_csv(V_fd_array, C_fd_array, filename="vfd_cfd_data.csv"):
        V_fd_flat = V_fd_array.flatten()
        C_fd_flat = C_fd_array.flatten()
        df = pd.DataFrame({
            "Pixel_Index": np.arange(len(V_fd_flat)),
            "V_fd (V)": V_fd_flat,
            "C_fd (F)": C_fd_flat
        })
        df.to_csv(filename, index=False)
        print(f"Saved V_fd and C_fd data to '{filename}'.")

    save_vfd_cfd_to_csv(V_fd_dynamic, C_fd_dynamic, "python_vfd_cfd_bright_v1.csv")

    # Plotting (corrected subplot indexing) with display
    fig = plt.figure(figsize=(20, 3))

    # Define GridSpec with 6 slots: 5 plots + 1 spacer
    gs = GridSpec(1, 6, width_ratios=[1, 1, 1, 1, 0.1, 1], wspace=0.1)

    # Define axes skipping the spacer (slot 3)
    axs = [
        fig.add_subplot(gs[0]),
        fig.add_subplot(gs[1]),
        fig.add_subplot(gs[2]),
        fig.add_subplot(gs[3]),
        fig.add_subplot(gs[5])
    ]

    titles = ['Over-exposed', 'Fixed $C_{\mathrm{FD}}$', 'ReRAM at FD', 'ΔCG (μV/e⁻)', 'ΔSNR (dB)']
    images = [ref_img, img_static, img_dynamic, CG_diff, SNR_diff]

    for i, (title, data) in enumerate(zip(titles, images)):
        ax = axs[i]
        if i < 3:
            ax.imshow(data, cmap='gray', vmin=0, vmax=255, aspect='auto')
        else:
            vmax = np.max(data)
            vmin = np.min(data)
            im = ax.imshow(data, cmap='bwr', vmin=vmin, vmax=vmax, aspect='auto')

            # Colorbar
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = plt.colorbar(im, cax=cax)
            cbar.locator = MaxNLocator(nbins=7)
            cbar.update_ticks()
            cbar.ax.tick_params(labelsize=12)
            for tick in cbar.ax.get_yticklabels():
                tick.set_fontweight('bold')

        ax.set_title(title, fontsize=12)
        ax.axis('off')

    # plt.suptitle(f"{scenario_label}: Sensor Response Comparison", fontsize=14)
    plt.tight_layout()
    plt.subplots_adjust(top=0.8)
    plt.savefig(pdf_name, bbox_inches='tight', dpi=300)
    plt.show()



    print("Max V_fd (Dynamic):", np.max(V_fd_dynamic))
    print("Min V_fd (Dynamic):", np.min(V_fd_dynamic))
    print("Max C_fd:", np.max(C_fd_dynamic))
    print("Min C_fd:", np.min(C_fd_dynamic))

# Run bright case with corrected model
process_and_plot(image_path, image_size, I_max_bright, "Bright Light", "python_bright_FINAL_redo.png", "bright")
