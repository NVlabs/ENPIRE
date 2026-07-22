# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Source this script to ensure RL_DATA_PATH and RL_KEYBOARD_DEVICE are set.
# If already set, this is a no-op. Otherwise prompt / auto-detect and
# persist the new value to ~/.bashrc.
# Usage: source assert_env_var.sh

_persist_to_bashrc() {
  local var="$1" val="$2" rc="$HOME/.bashrc"
  touch "$rc"
  sed -i "/^export ${var}=/d" "$rc"
  echo "export ${var}=\"${val}\"" >> "$rc"
  echo "Persisted ${var} to ${rc}"
}

if [ -z "$RL_DATA_PATH" ] || [ ! -d "$RL_DATA_PATH" ]; then
  [ -n "$RL_DATA_PATH" ] && echo "RL_DATA_PATH='$RL_DATA_PATH' does not exist."
  _options=()
  # Scan the current user's standard removable-media directory for mounted SSDs.
  _media_root="/media/${USER}"
  if [ -d "$_media_root" ]; then
    while IFS= read -r -d '' _mp; do
      _options+=("${_mp}/RL")
    done < <(find "$_media_root" -mindepth 1 -maxdepth 1 -type d -print0 2>/dev/null | sort -z)
  fi
  # Always include the home datasets fallback
  _home_opt="${HOME}/forge_datasets/RL"
  _already=0
  for _o in "${_options[@]}"; do [[ "$_o" == "$_home_opt" ]] && _already=1; done
  [[ "$_already" -eq 0 ]] && _options+=("$_home_opt")

  echo "RL_DATA_PATH not set. Choose a data saving path:"
  for _i in "${!_options[@]}"; do
    echo "  [$((${_i}+1))] ${_options[$_i]}"
  done
  echo "  [c] Custom path"

  while true; do
    read -e -p "Choice [1]: " _choice
    _choice="${_choice:-1}"
    if [[ "$_choice" == "c" || "$_choice" == "C" ]]; then
      read -e -p "  Enter custom path: " _custom
      export RL_DATA_PATH="$_custom"
      break
    elif [[ "$_choice" =~ ^[0-9]+$ ]] && \
         [ "$_choice" -ge 1 ] && [ "$_choice" -le "${#_options[@]}" ]; then
      export RL_DATA_PATH="${_options[$((_choice-1))]}"
      break
    else
      echo "  Invalid choice. Enter a number between 1 and ${#_options[@]}, or 'c'."
    fi
  done
  unset _options _home_opt _already _i _o _choice _custom _media_root
  _persist_to_bashrc RL_DATA_PATH "$RL_DATA_PATH"
fi
echo "RL_DATA_PATH=$RL_DATA_PATH"

if [ -z "$RL_KEYBOARD_DEVICE" ] || [ ! -e "$RL_KEYBOARD_DEVICE" ]; then
  [ -n "$RL_KEYBOARD_DEVICE" ] && echo "RL_KEYBOARD_DEVICE='$RL_KEYBOARD_DEVICE' does not exist."
  mapfile -t _kbds < <(ls /dev/input/by-id/*-event-kbd 2>/dev/null)
  if [ "${#_kbds[@]}" -eq 1 ]; then
    export RL_KEYBOARD_DEVICE="${_kbds[0]}"
    echo "Auto-detected keyboard: $RL_KEYBOARD_DEVICE"
  elif [ "${#_kbds[@]}" -gt 1 ]; then
    echo "Multiple keyboards found:"
    for _i in "${!_kbds[@]}"; do echo "  [$((_i+1))] ${_kbds[$_i]}"; done
    echo "  [c] Custom path"
    while true; do
      read -e -p "Choice [1]: " _choice
      _choice="${_choice:-1}"
      if [[ "$_choice" == "c" || "$_choice" == "C" ]]; then
        read -e -p "  Enter device path: " _custom
        export RL_KEYBOARD_DEVICE="$_custom"
        break
      elif [[ "$_choice" =~ ^[0-9]+$ ]] && \
           [ "$_choice" -ge 1 ] && [ "$_choice" -le "${#_kbds[@]}" ]; then
        export RL_KEYBOARD_DEVICE="${_kbds[$((_choice-1))]}"
        break
      else
        echo "  Invalid choice. Enter a number between 1 and ${#_kbds[@]}, or 'c'."
      fi
    done
    unset _choice _custom
  else
    echo "No keyboards found under /dev/input/by-id/*-event-kbd"
    read -e -p "Enter keyboard device path: " _custom
    export RL_KEYBOARD_DEVICE="$_custom"
    unset _custom
  fi
  _persist_to_bashrc RL_KEYBOARD_DEVICE "$RL_KEYBOARD_DEVICE"
  unset _kbds _i
fi
echo "RL_KEYBOARD_DEVICE=$RL_KEYBOARD_DEVICE"

if [ ! -r "$RL_KEYBOARD_DEVICE" ]; then
  echo "ERROR: cannot read $RL_KEYBOARD_DEVICE (Permission denied)."
  if id -nG | grep -qw input; then
    echo "Your shell session has not picked up the 'input' group yet."
    echo "Log out and log back in (or run: newgrp input) and re-run this script."
  else
    echo "User '$USER' is not in the 'input' group. Run:"
    echo "    sudo usermod -aG input $USER"
    echo "Then log out and log back in (or run: newgrp input) and re-run this script."
  fi
  unset -f _persist_to_bashrc
  return 1 2>/dev/null || exit 1
fi

unset -f _persist_to_bashrc
