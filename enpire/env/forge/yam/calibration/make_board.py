# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Render a printable ChArUco board matching this station's calibration config.

Generating the board from the same constants the detector uses removes the most
common calibration failure: a board printed at the wrong scale, or with a
different ArUco dictionary, still detects fine but reports poses in the wrong
units, so hand-eye silently converges to a wrong transform.

The PNG is rendered at a true physical scale, so it must be printed at 100% /
"actual size" with any fit-to-page scaling turned off.

    uv run python -m enpire.env.forge.yam.calibration.make_board --output board.png

Print it, measure one white square with calipers, and pass the measured value to
the calibrator if it differs from the nominal:

    uv run enpire station calibrate-all --station my-yam \\
      --square-length 0.0397 --marker-length 0.0298 --confirm-motion
"""

from __future__ import annotations

import argparse

from . import config


def render_board(
    *,
    squares_x: int,
    squares_y: int,
    square_length: float,
    marker_length: float,
    dictionary: str,
    dpi: int,
    margin_squares: float,
):
    """Render the board at ``dpi``, sized so a square measures ``square_length``."""
    import cv2

    if marker_length >= square_length:
        raise ValueError(
            f"marker_length ({marker_length}) must be smaller than square_length "
            f"({square_length}); the ArUco marker sits inside the checker square."
        )

    aruco_dict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary))
    board = cv2.aruco.CharucoBoard(
        (squares_x, squares_y), square_length, marker_length, aruco_dict
    )

    # metres -> inches -> pixels, so the printed square really is square_length.
    px_per_square = int(round(square_length / 0.0254 * dpi))
    size = (squares_x * px_per_square, squares_y * px_per_square)
    return board.generateImage(size, marginSize=int(px_per_square * margin_squares))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", default="charuco_board.png", help="PNG to write")
    parser.add_argument("--squares-x", type=int, default=config.SQUARES_X)
    parser.add_argument("--squares-y", type=int, default=config.SQUARES_Y)
    parser.add_argument("--square-length", type=float, default=config.SQUARE_LENGTH)
    parser.add_argument("--marker-length", type=float, default=config.MARKER_LENGTH)
    parser.add_argument("--dictionary", default=config.DICTIONARY)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--margin-squares",
        type=float,
        default=0.25,
        help="White quiet zone around the board, in squares. Detection needs one.",
    )
    args = parser.parse_args(argv)

    import cv2

    image = render_board(
        squares_x=args.squares_x,
        squares_y=args.squares_y,
        square_length=args.square_length,
        marker_length=args.marker_length,
        dictionary=args.dictionary,
        dpi=args.dpi,
        margin_squares=args.margin_squares,
    )
    if not cv2.imwrite(args.output, image):
        raise RuntimeError(f"failed to write {args.output}")

    width_mm = args.squares_x * args.square_length * 1000
    height_mm = args.squares_y * args.square_length * 1000
    print(f"Wrote {args.output} ({image.shape[1]}x{image.shape[0]} px @ {args.dpi} DPI)")
    print(
        f"Board: {args.squares_x}x{args.squares_y}, square "
        f"{args.square_length * 1000:.1f} mm, marker "
        f"{args.marker_length * 1000:.1f} mm, {args.dictionary}"
    )
    print(f"Printed size must measure {width_mm:.0f} x {height_mm:.0f} mm.")
    print("Print at 100% / actual size — disable 'fit to page'. Then verify with calipers.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
