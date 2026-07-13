"""Slice a 56/62-col AMP corpus down to a 30-col leg-only corpus.

Motivation
----------
Tasks that weld the arms (``K1_14dof_fixed_arms.urdf``) must not show the upper
body to the AMP discriminator. A frozen arm is a constant in every AMP frame
while the reference clips swing theirs, so the discriminator separates policy
from reference on that constant alone and the style reward collapses to zero —
taking the leg-style gradient with it.

Welding the arms changes neither the leg joint values nor the foot positions
(both are pure leg FK in the root frame), so the leg-only corpus is an exact
column slice of the existing corpus. No Isaac replay needed.

Layout
------
in  (56): joint_pos[0:22] joint_vel[22:44] EE[44:56]
          EE = left_hand(3) right_hand(3) left_foot(3) right_foot(3)
in  (62): the above + root_lin_vel[56:59] root_ang_vel[59:62]  (dropped)
out (30): leg_joint_pos[12] leg_joint_vel[12] left_foot(3) right_foot(3)

The 12 leg columns are emitted in Isaac Lab BFS order, matching
``LEG_JOINT_NAMES`` in ``soccer_stand_kick_amp/env_cfg.py`` (which pins the AMP
observation to that order via ``preserve_order=True``).

Usage
-----
    python scripts/make_leg_amp_corpus.py \
        --in-dir  booster_assets/motions/K1/motion_amp_expert/omni/kick \
        --out-dir booster_assets/motions/K1/motion_amp_expert/omni_legs/kick \
        --glob 'walk_kick*.txt' --rename walk_kick=stand_kick

The source clips are named ``walk_kick*`` but hold no walking gait: in each one a
single leg swings through (fore-aft range ~0.3 m) while the other stays planted
as the support foot, and neither foot lifts more than ~0.09 m. They are in-place
kicks, so the leg-only copies are renamed ``stand_kick*``. The source keeps its
name — ``soccer_kick_amp`` still globs it.
"""
from __future__ import annotations

import argparse
import glob
import json
import os

# Leg joint indices inside the 22-DOF joint block, in Isaac Lab BFS order.
# 3 Left_Hip_Pitch   4 Right_Hip_Pitch    8 Left_Hip_Roll    9 Right_Hip_Roll
# 12 Left_Hip_Yaw   13 Right_Hip_Yaw     16 Left_Knee_Pitch 17 Right_Knee_Pitch
# 18 Left_Ankle_Pitch 19 Right_Ankle_Pitch 20 Left_Ankle_Roll 21 Right_Ankle_Roll
LEG_IDX = [3, 4, 8, 9, 12, 13, 16, 17, 18, 19, 20, 21]

JOINT_POS_START = 0
JOINT_VEL_START = 22
EE_START = 44
FOOT_START = EE_START + 6  # skip left_hand(3) + right_hand(3)
FOOT_END = EE_START + 12

OUT_DIM = 2 * len(LEG_IDX) + 6  # 30


def slice_frame(frame: list[float]) -> list[float]:
    jp = [frame[JOINT_POS_START + i] for i in LEG_IDX]
    jv = [frame[JOINT_VEL_START + i] for i in LEG_IDX]
    feet = frame[FOOT_START:FOOT_END]
    return jp + jv + feet


def convert(src: str, dst: str) -> None:
    with open(src) as f:
        motion = json.load(f)

    frames = motion["Frames"]
    width = len(frames[0])
    if width not in (56, 62):
        raise ValueError(f"{src}: expected 56 or 62 columns, got {width}")

    out_frames = []
    for i, frame in enumerate(frames):
        if len(frame) != width:
            raise ValueError(f"{src}: frame {i} has {len(frame)} columns, expected {width}")
        out_frames.append(slice_frame(frame))

    motion["Frames"] = out_frames
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst, "w") as f:
        json.dump(motion, f)

    print(
        f"{os.path.basename(src)} -> {os.path.basename(dst)}: "
        f"{len(frames)} frames, {width} -> {OUT_DIM} cols"
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in-dir", required=True, help="directory of 56/62-col AMP txt clips")
    p.add_argument("--out-dir", required=True, help="directory to write the 30-col clips into")
    p.add_argument("--glob", default="*.txt", help="clip filename pattern (default: *.txt)")
    p.add_argument(
        "--rename",
        metavar="OLD=NEW",
        help="substring substitution applied to each output filename, e.g. walk_kick=stand_kick",
    )
    args = p.parse_args()

    srcs = sorted(glob.glob(os.path.join(args.in_dir, args.glob)))
    if not srcs:
        raise SystemExit(f"no clips matched {os.path.join(args.in_dir, args.glob)}")

    old, new = args.rename.split("=", 1) if args.rename else ("", "")

    for src in srcs:
        name = os.path.basename(src)
        if old:
            name = name.replace(old, new)
        convert(src, os.path.join(args.out_dir, name))

    print(f"\nwrote {len(srcs)} clips to {args.out_dir}")


if __name__ == "__main__":
    main()
