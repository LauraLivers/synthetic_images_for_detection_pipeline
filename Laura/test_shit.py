import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BACKGROUND_IMAGE = "robo_images/no_ladder/2b95fa3882b443218679c4d1241a7b16_back_5_March_2026_13-09.jpg"
REFERENCE_IMAGE  = "robo_images/ladder/5a2c1066f21b47abbee725417efe885c_front_5_March_2026_13-03.jpg"
PLACEMENTS_CSV = "MoBI_outputs/background9/placements.csv"
LADDER_INSTANCES_CSV = "MoBI_outputs/ladder_segmentations_real_size5/ladder_instances.csv"

def get_edge_lengths(corners):
    edges = []
    for i in range(4):
        a = corners[i]
        b = corners[(i + 1) % 4]
        edges.append((np.linalg.norm(b - a), i, (i + 1) % 4))
    return edges


def warp_to_slope(source_img, src_corners, dst_x1, dst_y1, dst_x2, dst_y2_l, dst_y2_r, canvas_W, canvas_H):
    edges = sorted(get_edge_lengths(src_corners), key=lambda e: e[0])
    _, ia, ib = edges[1]
    bl_src = src_corners[ia].astype(np.float64)
    br_src = src_corners[ib].astype(np.float64)
    if bl_src[0] > br_src[0]:
        bl_src, br_src = br_src, bl_src

    bl_dst = np.array([dst_x1, dst_y2_l], dtype=np.float64)
    br_dst = np.array([dst_x2, dst_y2_r], dtype=np.float64)

    src_vec = br_src - bl_src
    dst_vec = br_dst - bl_dst
    src_angle = np.arctan2(src_vec[1], src_vec[0])
    dst_angle = np.arctan2(dst_vec[1], dst_vec[0])
    angle = dst_angle - src_angle
    scale = np.linalg.norm(dst_vec) / np.linalg.norm(src_vec)

    cos_a, sin_a = np.cos(angle), np.sin(angle)
    R = scale * np.array([[cos_a, -sin_a], [sin_a, cos_a]])

    dst_corners = ((src_corners - bl_src) @ R.T) + bl_dst

    src_pts = src_corners.astype(np.float32)
    dst_pts = dst_corners.astype(np.float32)

    M, _ = cv2.findHomography(src_pts, dst_pts)
    if M is None:
        raise RuntimeError("findHomography failed")

    return cv2.warpPerspective(source_img, M, (canvas_W, canvas_H),
                               flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT,
                               borderValue=0)

def main():
    import random
    OUTPUT_DIR = "MoBI_outputs/batch_warp_test"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    bg_dir     = os.path.dirname(BACKGROUND_IMAGE)
    ladder_dir = os.path.dirname(REFERENCE_IMAGE)
    bg_images     = sorted([os.path.join(bg_dir, f) for f in os.listdir(bg_dir) if f.endswith(".jpg")])
    ladder_images = sorted([os.path.join(ladder_dir, f) for f in os.listdir(ladder_dir) if f.endswith(".jpg")])

    df_p = pd.read_csv(PLACEMENTS_CSV)
    df_l = pd.read_csv(LADDER_INSTANCES_CSV)

    for ref_img in ladder_images:
        for back_img in random.sample(bg_images, 3):
            bg_bgr = cv2.imread(back_img)
            bg_rgb = cv2.cvtColor(bg_bgr, cv2.COLOR_BGR2RGB)
            H, W = bg_rgb.shape[:2]

            name = os.path.splitext(os.path.basename(back_img))[0]
            all_placements = df_p[df_p["background_name"] == name]

            ref_bgr = cv2.imread(ref_img)
            ref_rgb = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2RGB)
            ref_row = df_l[df_l["image_path"].str.endswith(os.path.basename(ref_img))].iloc[0]
            ref_mask = np.load(ref_row["mask_path"]).astype(bool)
            ys, xs   = np.where(ref_mask)
            tight_y1, tight_y2 = ys.min(), ys.max() + 1
            tight_x1, tight_x2 = xs.min(), xs.max() + 1
            tight_rgb  = ref_rgb[tight_y1:tight_y2, tight_x1:tight_x2]
            mask_tight = ref_mask[tight_y1:tight_y2, tight_x1:tight_x2]

            ys_t, xs_t = np.where(mask_tight)
            pts    = np.stack([xs_t, ys_t], axis=1).astype(np.float64)
            center = pts.mean(axis=0)
            pts_c  = pts - center
            _, eigvecs = np.linalg.eigh(np.cov(pts_c.T))
            long_axis  = eigvecs[:, 1]
            short_axis = eigvecs[:, 0]
            proj_long  = pts_c @ long_axis
            proj_short = pts_c @ short_axis
            idx_tl = np.argmin(proj_long + proj_short)
            idx_tr = np.argmin(proj_long - proj_short)
            idx_br = np.argmax(proj_long + proj_short)
            idx_bl = np.argmax(proj_long - proj_short)
            corners_in_tight = np.array([
                pts[idx_tl], pts[idx_tr], pts[idx_br], pts[idx_bl]
            ], dtype=np.float32)

            colors = ["red", "cyan", "yellow"]

            for idx, (_, placement) in enumerate(all_placements.iterrows()):
                x1   = int(placement["x1"])
                y1   = int(placement["y1"])
                x2   = int(placement["x2"])
                y2_l = int(placement["y2_left"])
                y2_r = int(placement["y2_right"])

                ladder_ghost = warp_to_slope(tight_rgb, corners_in_tight, x1, y1, x2, y2_l, y2_r, W, H)
                warped_mask  = warp_to_slope(mask_tight.astype(np.uint8), corners_in_tight, x1, y1, x2, y2_l, y2_r, W, H).astype(bool)

                overlay = bg_rgb.copy()
                overlay[warped_mask] = ladder_ghost[warped_mask]

                fig, axes = plt.subplots(1, 3, figsize=(24, 8))
                fig.patch.set_facecolor("#111")

                axes[0].imshow(bg_rgb)
                for i, (_, row) in enumerate(all_placements.iterrows()):
                    c = colors[i % len(colors)]
                    rx1, ry1, rx2 = int(row["x1"]), int(row["y1"]), int(row["x2"])
                    ry2_l, ry2_r  = int(row["y2_left"]), int(row["y2_right"])
                    bpts = np.array([list(map(int, p.split(","))) for p in row["ground_boundary"].split(";")])
                    axes[0].plot(bpts[:, 0], bpts[:, 1], '-', color=c, lw=2, label=f"#{i} ground boundary")
                    axes[0].plot([rx1, rx2, rx2, rx1, rx1], [ry1, ry1, ry2_r, ry2_l, ry1], '--', color=c, lw=1)
                axes[0].set_title("Background Placement Zone", color="white")
                axes[0].legend()

                axes[1].imshow(tight_rgb)
                axes[1].scatter(corners_in_tight[:, 0], corners_in_tight[:, 1], c='cyan', s=80, marker='x', label="PCA Corners")
                sorted_by_y = corners_in_tight[corners_in_tight[:, 1].argsort()]
                bottom_two  = sorted_by_y[2:]
                axes[1].plot(bottom_two[:, 0], bottom_two[:, 1], 'r-', lw=3, label="Source Bottom Edge")
                axes[1].set_title("Tight Source Ladder", color="white")
                axes[1].legend()

                axes[2].imshow(overlay)
                axes[2].set_title(f"Placement #{idx} Warped Ghost Ladder", color="white")

                for ax in axes:
                    ax.axis("off")

                plt.tight_layout()
                bg_name  = os.path.splitext(os.path.basename(back_img))[0]
                ref_name = os.path.splitext(os.path.basename(ref_img))[0]
                out_path = f"{OUTPUT_DIR}/{bg_name}_{ref_name}_placement{idx}.jpg"                
                plt.savefig(out_path, dpi=150)
                plt.close()
                print(f"Saved {out_path}")

if __name__ == "__main__":
    main()