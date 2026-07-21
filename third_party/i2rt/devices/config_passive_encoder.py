from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import IntEnum
import struct
import time
from typing import Literal
from can import BusABC, Message
import can
import click

ALL_DEVICE = 0xFF
"""All devices for broadcasting requests."""


class EEPROMField(IntEnum):
    """The EEPROM field."""

    ZPOS_H = 9
    ZPOS_L = 10


@dataclass
class EncoderReport:
    """The encoder report."""

    device: int
    """The device number, uint8."""
    position: int
    """The position in encoder counts, int16."""
    velocity: int
    """The velocity in encoder, int16."""
    inputs: int
    """The inputs, bit by bit,uint8."""


@dataclass
class VersionReply:
    """The version reply."""

    device: int
    """The device number, uint8."""
    major: int
    """The major version, uint8."""
    minor: int
    """The minor version, uint8."""
    patch: int
    """The patch version, uint8."""


class EncoderCanID(IntEnum):
    """The CAN ID for the encoder."""

    REQ = 0x50E
    """The CAN ID for sending a request/reply."""
    REPORT = 0x50F
    """The CAN ID for reporting the encoder status."""
    EVENT = 0x510
    """The CAN ID for reporting the encoder event, e.g. button press."""


class Encoder:
    """The encoder driver."""

    REQ_ZPOS = 0x00
    """Request setting the zero position."""
    REQ_FREQ = 0x01
    """Request setting the report frequency."""
    REQ_REPORT = 0x02
    """Request getting the encoder report."""
    REQ_VERSION = 0x03
    """Request getting the version."""
    REQ_ADC_FREQ = 0x04
    """Request setting the ADC frequency."""
    REQ_DIO_EVENT = 0x05
    """Request toggling the digital IO event report."""
    REQ_READINGS = 0x06
    """Request reading the analog and digital values."""
    REQ_GET_EEPROM = 0x07
    """Request reading EEPROM field by index."""
    REQ_RESTART = 0x0F
    """Restart encoder."""

    def __init__(self, bus: BusABC):
        self.bus = bus
        self.bus.set_filters(
            [
                {"can_id": EncoderCanID.REPORT, "can_mask": 0x7FF},
                {"can_id": EncoderCanID.EVENT, "can_mask": 0x7FF},
                {"can_id": EncoderCanID.REQ, "can_mask": 0x7FF},
            ]
        )

    def reset_zero_position(self, device: int = ALL_DEVICE) -> None:
        """Set current position as the zero position."""
        assert 0 <= device <= 255, "Device must be between 0 and 255"
        message = Message(
            arbitration_id=EncoderCanID.REQ,
            data=[device, self.REQ_ZPOS],
            is_extended_id=False,
        )
        self.bus.send(message)

    def set_report_frequency(self, frequency: int, device: int = ALL_DEVICE) -> None:
        """Set the report frequency, 0 for passive mode."""
        assert 0 <= frequency <= 255, "Report frequency must be between 0 and 255"
        assert 0 <= device <= 255, "Device must be between 0 and 255"
        message = Message(
            arbitration_id=EncoderCanID.REQ,
            data=[device, self.REQ_FREQ, frequency],
            is_extended_id=False,
        )
        self.bus.send(message)

    def set_adc_frequency(self, frequency: int, device: int = ALL_DEVICE) -> None:
        """Set the ADC sampling frequency."""
        assert 0 <= frequency <= 255, "ADC frequency must be between 0 and 255"
        assert 0 <= device <= 255, "Device must be between 0 and 255"
        message = Message(
            arbitration_id=EncoderCanID.REQ,
            data=[device, self.REQ_ADC_FREQ, frequency],
            is_extended_id=False,
        )
        self.bus.send(message)

    def get_encoder_report(
        self, device: int = ALL_DEVICE, timeout: float | None = None
    ) -> EncoderReport | None:
        """Get the encoder report."""
        assert 0 <= device <= 255, "Device must be between 0 and 255"
        message = Message(
            arbitration_id=EncoderCanID.REQ,
            data=[device, self.REQ_REPORT],
            is_extended_id=False,
        )
        self.bus.send(message)
        return self.wait_for_report(device, timeout)

    def wait_for(
        self,
        message_id: Literal[EncoderCanID.REPORT, EncoderCanID.EVENT],
        device: int = ALL_DEVICE,
        timeout: float | None = None,
    ) -> EncoderReport | None:
        """Wait for a report."""
        assert 0 <= device <= 255, "Device must be between 0 and 255"
        start_time = time.time()
        while True:
            message = self.bus.recv(timeout=timeout)
            if message and message.arbitration_id == message_id:
                assert len(message.data) == 6, "Report must be 6 bytes"
                from_device = message.data[0]
                if device != ALL_DEVICE and from_device != device:
                    continue
                position = struct.unpack(">h", message.data[1:3])[0]
                velocity = struct.unpack(">h", message.data[3:5])[0]
                inputs = message.data[5]
                return EncoderReport(from_device, position, velocity, inputs)
            if message is None or (
                timeout is not None and time.time() - start_time > timeout
            ):
                return None

    def wait_for_event(
        self, device: int = ALL_DEVICE, timeout: float | None = None
    ) -> EncoderReport | None:
        """Wait for an event."""
        return self.wait_for(EncoderCanID.EVENT, device, timeout)

    def wait_for_report(
        self, device: int = ALL_DEVICE, timeout: float | None = None
    ) -> EncoderReport | None:
        """Wait for a report."""
        return self.wait_for(EncoderCanID.REPORT, device, timeout)

    def get_version(
        self, device: int = ALL_DEVICE, timeout: float | None = None
    ) -> VersionReply | None:
        """Get the version."""
        assert 0 <= device <= 255, "Device must be between 0 and 255"
        req = Message(
            arbitration_id=EncoderCanID.REQ,
            data=[device, self.REQ_VERSION],
            is_extended_id=False,
        )
        self.bus.send(req)
        start_time = time.time()
        while True:
            message = self.bus.recv(timeout=timeout)
            if message and message.arbitration_id == EncoderCanID.REQ:
                if len(message.data) != 5:
                    continue
                from_device = message.data[0]
                if device != ALL_DEVICE and from_device != device:
                    continue
                cmd = message.data[1]
                if cmd != (self.REQ_VERSION | (1 << 7)):
                    continue
                return VersionReply(
                    device=from_device,
                    major=message.data[2],
                    minor=message.data[3],
                    patch=message.data[4],
                )
            if timeout is not None and time.time() - start_time > timeout:
                return None

    def toggle_digital_io_event_report(
        self, device: int = ALL_DEVICE, io_mask: int = 0
    ) -> None:
        """Toggle the digital IO event report."""
        assert 0 <= device <= 255, "Device must be between 0 and 255"
        assert 0 <= io_mask <= 255, "IO mask must be between 0 and 255"
        req = Message(
            arbitration_id=EncoderCanID.REQ,
            data=[device, self.REQ_DIO_EVENT, io_mask],
            is_extended_id=False,
        )
        self.bus.send(req)

    def get_readings(
        self,
        device: int = ALL_DEVICE,
        analog_index: int = 0,
        timeout: float | None = None,
    ) -> tuple[int, int] | None:
        """Get the readings."""
        assert 0 <= device <= 255, "Device must be between 0 and 255"
        assert 0 <= analog_index <= 3, "Analog index must be between 0 and 3"
        req = Message(
            arbitration_id=EncoderCanID.REQ,
            data=[device, self.REQ_READINGS, analog_index],
            is_extended_id=False,
        )
        self.bus.send(req)
        start_time = time.time()
        while True:
            message = self.bus.recv(timeout=timeout)
            if message and message.arbitration_id == EncoderCanID.REQ:
                if len(message.data) != 5:
                    continue
                from_device = message.data[0]
                if device != ALL_DEVICE and from_device != device:
                    continue
                cmd = message.data[1]
                if cmd != (self.REQ_READINGS | (1 << 7)):
                    continue
                analog_value = struct.unpack(">h", message.data[2:4])[0]
                digital_value = message.data[4]
                return analog_value, digital_value
            if timeout is not None and time.time() - start_time > timeout:
                return None

    def read_eeprom_field(
        self, offset: int, device: int = ALL_DEVICE, timeout: float | None = None
    ) -> int | None:
        """Read EEPROM by offset address. Returns byte value or None."""
        assert 0 <= device <= 255, "Device must be between 0 and 255"
        assert 0 <= offset < 27, f"Offset must be 0-26"

        req = Message(
            arbitration_id=EncoderCanID.REQ,
            data=[device, self.REQ_GET_EEPROM, offset],
            is_extended_id=False,
        )
        self.bus.send(req)

        # Wait for response using existing readings format
        start_time = time.time()
        while True:
            message = self.bus.recv(timeout=timeout)
            if message and message.arbitration_id == EncoderCanID.REQ:
                if len(message.data) != 5:
                    continue
                from_device = message.data[0]
                if device != ALL_DEVICE and from_device != device:
                    continue
                cmd = message.data[1]
                if cmd != (
                    self.REQ_READINGS | (1 << 7)
                ):  # Response uses READINGS format
                    continue
                value = struct.unpack(">h", message.data[2:4])[0]
                return value & 0xFF  # Return as byte
            if timeout is not None and time.time() - start_time > timeout:
                return None

    def restart(self, device: int = ALL_DEVICE) -> None:
        """Restart encoder."""
        assert 0 <= device <= 255, "Device must be between 0 and 255"
        req = Message(
            arbitration_id=EncoderCanID.REQ,
            data=[device, self.REQ_RESTART],
            is_extended_id=False,
        )
        self.bus.send(req)


@click.group()
@click.option("--bus", type=str, default="can0", show_default=True)
@click.option("--device", type=int, default=ALL_DEVICE, show_default=True)
@click.option("--bitrate", type=int, default=1000000, show_default=True)
@click.pass_context
def cli(ctx, bus: str, device: int, bitrate: int) -> None:
    can_bus = can.interface.Bus(interface="socketcan", channel=bus, bitrate=bitrate)
    encoder = Encoder(can_bus)
    ctx.ensure_object(dict)
    ctx.obj["encoder"] = encoder
    ctx.obj["device"] = device
    ctx.call_on_close(can_bus.shutdown)


@cli.command()
@click.pass_context
def reset_zero_position(ctx) -> None:
    encoder: Encoder = ctx.obj["encoder"]
    device: int = ctx.obj["device"]
    encoder.reset_zero_position(device)


@cli.command()
@click.pass_context
@click.argument("frequency", type=int, default=0)
def set_report_frequency(ctx, frequency: int) -> None:
    encoder: Encoder = ctx.obj["encoder"]
    device: int = ctx.obj["device"]
    encoder.set_report_frequency(frequency, device)


@cli.command()
@click.pass_context
@click.argument("frequency", type=int)
def set_adc_frequency(ctx, frequency: int) -> None:
    encoder: Encoder = ctx.obj["encoder"]
    device: int = ctx.obj["device"]
    encoder.set_adc_frequency(frequency, device)


@cli.command()
@click.option("--timeout", type=float, default=None, show_default=True)
@click.pass_context
def get_report(ctx, timeout: float | None = None) -> None:
    encoder: Encoder = ctx.obj["encoder"]
    device: int = ctx.obj["device"]
    report: EncoderReport | None = encoder.get_encoder_report(device, timeout)
    if report is None:
        print("No report")
    else:
        print(report)


@cli.command()
@click.option("--timeout", type=float, default=None, show_default=True)
@click.pass_context
def wait_for_event(ctx, timeout: float | None = None) -> None:
    encoder: Encoder = ctx.obj["encoder"]
    device: int = ctx.obj["device"]
    event: EncoderReport | None = encoder.wait_for_event(device, timeout)
    if event is None:
        print("No event")
    else:
        print(event)


@cli.command()
@click.option("--timeout", type=float, default=None, show_default=True)
@click.pass_context
def wait_for_report(ctx, timeout: float | None = None) -> None:
    encoder: Encoder = ctx.obj["encoder"]
    device: int = ctx.obj["device"]
    report: EncoderReport | None = encoder.wait_for_report(device, timeout)
    if report is None:
        print("No report")
    else:
        print(report)


@cli.command()
@click.option("--timeout", type=float, default=None, show_default=True)
@click.pass_context
def wait_for_event_or_report(ctx, timeout: float | None = None) -> None:
    encoder: Encoder = ctx.obj["encoder"]
    device: int = ctx.obj["device"]
    executor = ThreadPoolExecutor(max_workers=2)
    event_future = executor.submit(encoder.wait_for_event, device, timeout)
    report_future = executor.submit(encoder.wait_for_report, device, timeout)
    while not event_future.done() or not report_future.done():
        time.sleep(0.1)
    if not event_future.done():
        event_future.cancel()
    if not report_future.done():
        report_future.cancel()


@cli.command()
@click.option("--timeout", type=float, default=1, show_default=True)
@click.pass_context
def get_version(ctx, timeout: float) -> None:
    encoder: Encoder = ctx.obj["encoder"]
    device: int = ctx.obj["device"]
    version_reply = encoder.get_version(device, timeout)
    if version_reply is None:
        print("No version")
    else:
        print(version_reply)


@cli.command()
@click.argument("io_mask", type=int, required=True)
@click.pass_context
def toggle_digital_io_event(ctx, io_mask: int) -> None:
    """Toggle the digital IO events, io_mask is a bit mask of the IOs, 0 to disable all IOs."""
    encoder: Encoder = ctx.obj["encoder"]
    device: int = ctx.obj["device"]
    encoder.toggle_digital_io_event_report(device, io_mask)


@cli.command()
@click.argument("analog_index", type=int, required=True)
@click.pass_context
def get_readings(ctx, analog_index: int) -> None:
    encoder: Encoder = ctx.obj["encoder"]
    device: int = ctx.obj["device"]
    assert 0 <= analog_index <= 3, "Analog index must be between 0 and 3"
    readings = encoder.get_readings(device, analog_index)
    if readings is None:
        print("No readings")
    else:
        print(f"Analog: 0x{readings[0]:04x}, Digital: 0x{readings[1]:02x}")


@cli.command()
@click.argument("offset", type=int, required=True)
@click.pass_context
def read_eeprom(ctx, offset: int) -> None:
    """Read EEPROM field by offset."""
    encoder: Encoder = ctx.obj["encoder"]
    device: int = ctx.obj["device"]

    # EEPROM offset and field name mapping
    EEPROM_FIELDS = {
        0: "magic_h",
        1: "magic_l",
        2: "can_ext",
        3: "can_id_0",
        4: "can_id_1",
        5: "can_id_2",
        6: "can_id_3",
        7: "device",
        8: "adc_freq",
        9: "zpos_h",
        10: "zpos_l",
        11: "mpos_h",
        12: "mpos_l",
        13: "mang_h",
        14: "mang_l",
        15: "filters_begin",
        23: "dir",
        24: "threshold_steps",
        25: "report_freq",
        26: "dio_report_reverse",
    }

    field_name = EEPROM_FIELDS.get(offset, "unknown")
    print(f"Reading EEPROM field '{field_name}' at offset {offset}...")
    value = encoder.read_eeprom_field(offset, device, timeout=1.0)

    if value is None:
        print("No response from device")
    else:
        print(f"Value: {value} (0x{value:02X})")


@cli.command()
@click.pass_context
def read_eeprom_zpos(ctx) -> None:
    encoder: Encoder = ctx.obj["encoder"]
    device: int = ctx.obj["device"]
    zpos_bytes_0 = encoder.read_eeprom_field(EEPROMField.ZPOS_H, device, timeout=1.0)
    zpos_bytes_1 = encoder.read_eeprom_field(EEPROMField.ZPOS_L, device, timeout=1.0)
    if zpos_bytes_0 is None or zpos_bytes_1 is None:
        print("No response from device")
    else:
        zpos = struct.unpack("<h", bytes([zpos_bytes_0, zpos_bytes_1]))[
            0
        ]  # little endian
        print(f"ZPOS: 0x{zpos:04x}")


@cli.command()
@click.pass_context
def restart(ctx) -> None:
    """Restart the encoder."""
    encoder: Encoder = ctx.obj["encoder"]
    device: int = ctx.obj["device"]
    encoder.restart(device)


if __name__ == "__main__":
    cli()
