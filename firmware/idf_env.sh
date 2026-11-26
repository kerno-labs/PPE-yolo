# Source this file before running idf.py menuconfig / build / flash / monitor:
#   source firmware/idf_env.sh
export IDF_PATH="${IDF_PATH:-$HOME/esp/esp-idf}"
export IDF_PYTHON_ENV_PATH="${IDF_PYTHON_ENV_PATH:-$HOME/.espressif/python_env/idf5.5_py3.11_env}"
export ESP_ROM_ELF_DIR="$(ls -d "$HOME"/.espressif/tools/esp-rom-elfs/*/ 2>/dev/null | head -n1)"
export IDF_SKIP_CHECK_SUBMODULES=1

_XTENSA_BIN="$(ls -d "$HOME"/.espressif/tools/xtensa-esp-elf/*/xtensa-esp-elf/bin 2>/dev/null | head -n1)"
export PATH="$IDF_PYTHON_ENV_PATH/bin:$_XTENSA_BIN:$IDF_PATH/tools:/opt/homebrew/bin:$PATH"
unset _XTENSA_BIN
