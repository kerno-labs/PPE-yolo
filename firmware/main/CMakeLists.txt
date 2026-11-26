cmake_minimum_required(VERSION 3.16)

idf_component_register(
    SRCS
        "src/app_main.cpp"
        "src/violation_engine.cpp"
        "src/camera_capture.cpp"
        "src/display_ui.cpp"
        "src/audio_alert.cpp"
        "src/wecom_notifier.cpp"
    INCLUDE_DIRS
        "include"
    EMBED_FILES
        "assets/alert_zh.pcm"
    REQUIRES
        nvs_flash
        esp_wifi
        esp_event
        esp_netif
        esp_http_client
        esp_timer
        mbedtls
        esp-dl
        yolo26_ppe
)

# Resolve esp-dl component directory for 16-byte aligned .espdl embedding
idf_build_get_property(component_targets __COMPONENT_TARGETS)
if("___idf_espressif__esp-dl" IN_LIST component_targets)
    idf_component_get_property(espdl_dir espressif__esp-dl COMPONENT_DIR)
else()
    idf_component_get_property(espdl_dir esp-dl COMPONENT_DIR)
endif()

set(MODEL_FILE "${CMAKE_CURRENT_LIST_DIR}/models/yolo26n_ppe_320_s8_s3.espdl")
if(NOT EXISTS "${MODEL_FILE}")
    message(FATAL_ERROR "Missing quantized ESP-DL model at ${MODEL_FILE}. Run ml/quantize_espdl.py first!")
endif()

set(cmake_dir "${espdl_dir}/fbs_loader/cmake")
include("${cmake_dir}/utilities.cmake")
target_add_aligned_binary_data(${COMPONENT_LIB} "${MODEL_FILE}" BINARY)
