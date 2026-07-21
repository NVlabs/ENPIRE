# YAM Handle Basic

## Reading Handle Output

To read the encoder output, run the following command:

```bash
python scripts/read_encoder.py --channel $CAN_CHANNEL
```

You should see output similar to this:

```bash
[PassiveEncoderInfo(id=1294, position=np.float64(0.004382802251101832), velocity=0.0, io_inputs=[0, 0])]
```

**Key Information:**
- `position` represents the trigger position, which will later be mapped to the gripper position
- The handle has two reserved buttons that can be mapped to desired functions
- When the trigger is not pulled, the position reading should be zero
- When the trigger is fully pulled, the position reading should be near 1.0

## Resetting Encoder's Zero Position

If the magnet inside the encoder has shifted or after repairs, you may need to calibrate the encoder.

To reset the encoder's zero position, run:

```bash
python devices/config_passive_encoder.py --bus $CAN_CHANNEL reset-zero-position
```

This command sends a reset signal to the encoder board. After running this command, you can verify the reset was successful by running `read_encoder.py` again to check the readings.
