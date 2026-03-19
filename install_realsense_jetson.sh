#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# Intel RealSense SDK (librealsense) 설치 스크립트 — Jetson AGX Orin
#
# 사용법:
#   chmod +x install_realsense_jetson.sh
#   ./install_realsense_jetson.sh
#
# 옵션 (환경변수로 변경 가능):
#   REALSENSE_TAG    : librealsense 태그/브랜치  (기본: v2.56.3)
#   INSTALL_DIR      : 소스 클론 디렉토리        (기본: $HOME/librealsense)
#   PYTHON_BIN       : Python 실행파일           (기본: python3)
#   BUILD_JOBS       : 병렬 빌드 수              (기본: $(nproc))
###############################################################################

REALSENSE_TAG="${REALSENSE_TAG:-v2.56.3}"
INSTALL_DIR="${INSTALL_DIR:-$HOME/librealsense}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
BUILD_JOBS="${BUILD_JOBS:-$(nproc)}"

PYTHON_VERSION=$("${PYTHON_BIN}" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")

echo "============================================="
echo " RealSense SDK 설치 시작"
echo "  태그:        ${REALSENSE_TAG}"
echo "  소스 경로:   ${INSTALL_DIR}"
echo "  Python:      ${PYTHON_BIN} (${PYTHON_VERSION})"
echo "  빌드 스레드: ${BUILD_JOBS}"
echo "============================================="

# ── 1. 의존성 설치 ──
echo "[1/6] 시스템 패키지 설치..."
sudo apt-get update
sudo apt-get install -y \
    git cmake build-essential pkg-config \
    libssl-dev libusb-1.0-0-dev \
    libgtk-3-dev libglfw3-dev \
    libgl1-mesa-dev libglu1-mesa-dev

# ── 2. 소스 클론 ──
echo "[2/6] librealsense 소스 클론..."
if [ -d "${INSTALL_DIR}" ]; then
    echo "  이미 존재: ${INSTALL_DIR} — pull & checkout"
    cd "${INSTALL_DIR}"
    git fetch --tags
    git checkout "${REALSENSE_TAG}"
else
    git clone --depth 1 --branch "${REALSENSE_TAG}" \
        https://github.com/IntelRealSense/librealsense.git "${INSTALL_DIR}"
    cd "${INSTALL_DIR}"
fi

# ── 3. CMakeLists.txt 호환성 패치 (CMake 3.27+ 대응) ──
# 오래된 cmake_minimum_required(VERSION 2.8.3) → 3.5로 상향
if grep -q 'cmake_minimum_required.*VERSION.*2\.' CMakeLists.txt 2>/dev/null; then
    echo "  CMakeLists.txt cmake_minimum_required 패치 적용..."
    sed -i 's/cmake_minimum_required(\s*VERSION\s*[0-9.]\+/cmake_minimum_required(VERSION 3.5/' CMakeLists.txt
fi

# ── 4. 빌드 ──
echo "[3/6] CMake 빌드 (Python 바인딩 포함)..."
mkdir -p build && cd build
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_PYTHON_BINDINGS=true \
    -DPYTHON_EXECUTABLE="$(which "${PYTHON_BIN}")" \
    -DBUILD_EXAMPLES=false \
    -DBUILD_GRAPHICAL_EXAMPLES=false \
    -DFORCE_RSUSB_BACKEND=true

echo "[4/6] 컴파일 중... (j${BUILD_JOBS})"
make -j"${BUILD_JOBS}"

echo "[5/6] 시스템 설치..."
sudo make install

# ── 4. udev 규칙 ──
echo "[6/6] udev 규칙 설정..."
if [ -f "${INSTALL_DIR}/config/99-realsense-libusb.rules" ]; then
    sudo cp "${INSTALL_DIR}/config/99-realsense-libusb.rules" /etc/udev/rules.d/
    sudo udevadm control --reload-rules
    sudo udevadm trigger
    echo "  udev 규칙 설치 완료"
else
    echo "  udev 규칙 파일을 찾을 수 없음 (수동 설치 필요)"
fi

# ── 5. PYTHONPATH 등록 ──
RS_PYPATH=$(find /usr/local/lib -type d -name "pyrealsense2" 2>/dev/null | head -1)
if [ -z "${RS_PYPATH}" ]; then
    RS_PYPATH="/usr/local/lib/python${PYTHON_VERSION}/pyrealsense2"
fi
RS_PARENT=$(dirname "${RS_PYPATH}")

SHELL_RC="$HOME/.bashrc"
if ! grep -q "pyrealsense2" "${SHELL_RC}" 2>/dev/null; then
    echo "" >> "${SHELL_RC}"
    echo "# Intel RealSense Python 바인딩" >> "${SHELL_RC}"
    echo "export PYTHONPATH=\"${RS_PARENT}:\${PYTHONPATH:-}\"" >> "${SHELL_RC}"
    echo "  PYTHONPATH 추가됨 → ${RS_PARENT}"
else
    echo "  PYTHONPATH에 pyrealsense2 이미 등록됨"
fi

# ── 6. 검증 ──
echo ""
echo "============================================="
export PYTHONPATH="${RS_PARENT}:${PYTHONPATH:-}"
if "${PYTHON_BIN}" -c "import pyrealsense2 as rs; print(f'pyrealsense2 {rs.__version__} 설치 성공')" 2>/dev/null; then
    echo "============================================="
    echo " 설치 완료!"
else
    echo " Python import 실패 — PYTHONPATH 확인 필요"
    echo "   ${RS_PARENT} 에 pyrealsense2 모듈이 있는지 확인하세요"
    echo "============================================="
    exit 1
fi
