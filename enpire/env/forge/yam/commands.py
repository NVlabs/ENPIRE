"""CLI handlers for station-local configuration."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, replace

from enpire.env.forge.yam.doctor import check_station, station_ready
from enpire.env.forge.yam.registration.rules import (
    discover_registration,
    write_registration_files,
)
from enpire.env.forge.yam.station import StationProfile, load_station, save_station


def station_init(args: argparse.Namespace) -> int:
    path = save_station(StationProfile(station_id=args.station), args.config_root)
    print(path)
    return 0


def station_register(args: argparse.Namespace) -> int:
    registration = discover_registration(
        args.station,
        can=not args.no_can,
        buttons=not args.no_buttons,
        cameras=not args.no_cameras,
        top_camera=args.top_camera,
    )
    try:
        station = load_station(args.station, args.config_root)
    except FileNotFoundError:
        station = StationProfile(station_id=args.station)
    station = replace(station, devices=dict(registration.roles))
    profile_path = save_station(station, args.config_root)
    rules, aliases = write_registration_files(registration, station.data_dir / "registration")
    print(f"profile: {profile_path}")
    print(f"rules: {rules}")
    print(f"camera aliases: {aliases}")
    print("No system files were installed. Review the generated files first.")
    return 0


def station_show(args: argparse.Namespace) -> int:
    profile = load_station(args.station, args.config_root)
    print(json.dumps(asdict(profile), indent=2, sort_keys=True))
    return 0


def station_doctor(args: argparse.Namespace) -> int:
    profile = load_station(args.station, args.config_root)
    checks = check_station(profile)
    if args.json:
        print(json.dumps([asdict(check) for check in checks], indent=2))
    else:
        for check in checks:
            mark = "PASS" if check.passed else "FAIL"
            print(f"{mark:<4} {check.name:<28} {check.detail}")
    return 0 if station_ready(checks) else 1


def station_gravcomp(args: argparse.Namespace) -> int:
    """Run the preserved Forge gravity-compensation viewer lazily."""

    os.environ["ENPIRE_STATION"] = args.station
    argv = [f"--{args.side_selection}", "--camera", args.camera]
    if args.no_launch_server:
        argv.append("--no-launch-server")
    if args.force_launch_server:
        argv.append("--force-launch-server")
    if args.no_gui:
        argv.append("--no-gui")
    if args.duration:
        argv.extend(["--duration", str(args.duration)])
    if args.confirm_motion:
        argv.append("--confirm-motion")

    from enpire.env.forge.tools.debug.yam_gravcomp_viewer import main

    return int(main(argv))


def station_calibrate(args: argparse.Namespace) -> int:
    os.environ["ENPIRE_STATION"] = args.station
    if args.model_root is not None:
        os.environ["ENPIRE_YAM_MODEL_ROOT"] = args.model_root
    if args.output_xml is not None:
        os.environ["ENPIRE_YAM_CALIBRATED_XML_OUTPUT"] = args.output_xml

    from enpire.env.forge.yam.calibration.run import run_calibration

    return run_calibration(
        camera=args.camera,
        resolution=args.resolution,
        launch_server=not args.no_launch_server,
        confirm_motion=args.confirm_motion,
    )


def station_validate_calibration(args: argparse.Namespace) -> int:
    from enpire.env.forge.yam.calibration.bundle import (
        load_calibration_record,
        validate_calibration_record,
    )

    record = load_calibration_record(args.calibration_json)
    errors = validate_calibration_record(
        record,
        max_translation_rms_mm=args.max_translation_rms_mm,
        max_rotation_rms_deg=args.max_rotation_rms_deg,
    )
    if errors:
        for error in errors:
            print(f"FAIL {error}")
        return 1
    print(f"PASS {record.camera_name}: calibration geometry and residuals are valid")
    return 0


def station_calibrate_all(args: argparse.Namespace) -> int:
    os.environ["ENPIRE_STATION"] = args.station
    if args.model_root is not None:
        os.environ["ENPIRE_YAM_MODEL_ROOT"] = args.model_root
    else:
        # Always pin to gear-enpire's bundled models so stale env vars
        # from old forge/yam-calibration installs don't redirect the path.
        from enpire.env.forge.paths import forge_path
        os.environ["ENPIRE_YAM_MODEL_ROOT"] = str(forge_path("robot", "models", "station"))
    if args.output_xml is not None:
        os.environ["ENPIRE_YAM_CALIBRATED_XML_OUTPUT"] = args.output_xml
    from enpire.env.forge.yam.calibration.launch import launch
    return launch(
        confirm_motion=args.confirm_motion,
        station=args.station,
        resolution=args.resolution,
    )


def add_station_parser(commands: argparse._SubParsersAction) -> None:
    station = commands.add_parser("station", help="Configure and inspect a YAM station")
    station_commands = station.add_subparsers(dest="station_command", required=True)

    init = station_commands.add_parser("init", help="Create an external station profile")
    init.add_argument("--station", required=True)
    init.add_argument("--config-root", default=None)
    init.set_defaults(handler=station_init)

    register = station_commands.add_parser(
        "register", help="Identify devices and render registration files without sudo"
    )
    register.add_argument("--station", required=True)
    register.add_argument("--config-root", default=None)
    register.add_argument("--no-can", action="store_true")
    register.add_argument("--no-buttons", action="store_true")
    register.add_argument("--no-cameras", action="store_true")
    register.add_argument("--top-camera", action="store_true")
    register.set_defaults(handler=station_register)

    show = station_commands.add_parser("show", help="Show an external station profile")
    show.add_argument("--station", required=True)
    show.add_argument("--config-root", default=None)
    show.set_defaults(handler=station_show)

    doctor = station_commands.add_parser("doctor", help="Run read-only station checks")
    doctor.add_argument("--station", required=True)
    doctor.add_argument("--config-root", default=None)
    doctor.add_argument("--json", action="store_true")
    doctor.set_defaults(handler=station_doctor)

    gravcomp = station_commands.add_parser(
        "gravcomp", help="Run the source-faithful YAM gravity-compensation viewer"
    )
    gravcomp.add_argument("--station", required=True)
    side = gravcomp.add_mutually_exclusive_group()
    side.add_argument("--left", dest="side_selection", action="store_const", const="left")
    side.add_argument("--right", dest="side_selection", action="store_const", const="right")
    side.add_argument("--both", dest="side_selection", action="store_const", const="both")
    gravcomp.set_defaults(side_selection="left")
    gravcomp.add_argument(
        "--camera", choices=("left", "right", "top", "both", "none"), default="both"
    )
    gravcomp.add_argument("--no-launch-server", action="store_true")
    gravcomp.add_argument("--force-launch-server", action="store_true")
    gravcomp.add_argument("--no-gui", action="store_true")
    gravcomp.add_argument("--duration", type=float, default=0.0)
    gravcomp.add_argument("--confirm-motion", action="store_true")
    gravcomp.set_defaults(handler=station_gravcomp)

    calibrate = station_commands.add_parser(
        "calibrate", help="Run one pass of the integrated YAM ChArUco calibration"
    )
    calibrate.add_argument("--station", required=True)
    calibrate.add_argument("--camera", choices=("top", "left_wrist", "right_wrist"), default="top")
    calibrate.add_argument("--resolution", default=None, metavar="WIDTHxHEIGHT")
    calibrate.add_argument("--model-root", default=None)
    calibrate.add_argument("--output-xml", default=None)
    calibrate.add_argument("--no-launch-server", action="store_true")
    calibrate.add_argument("--confirm-motion", action="store_true")
    calibrate.set_defaults(handler=station_calibrate)

    calibrate_all = station_commands.add_parser(
        "calibrate-all",
        help="Start arm servers in tmux and run all three camera calibrations in sequence",
    )
    calibrate_all.add_argument("--station", required=True)
    calibrate_all.add_argument("--resolution", default=None, metavar="WIDTHxHEIGHT")
    calibrate_all.add_argument("--model-root", default=None)
    calibrate_all.add_argument("--output-xml", default=None)
    calibrate_all.add_argument("--confirm-motion", action="store_true")
    calibrate_all.set_defaults(handler=station_calibrate_all)

    validate = station_commands.add_parser(
        "validate-calibration", help="Validate an emitted calibration.json without hardware"
    )
    validate.add_argument("calibration_json")
    validate.add_argument("--max-translation-rms-mm", type=float, default=20.0)
    validate.add_argument("--max-rotation-rms-deg", type=float, default=5.0)
    validate.set_defaults(handler=station_validate_calibration)
