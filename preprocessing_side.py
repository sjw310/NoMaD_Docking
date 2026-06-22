import os
import cv2
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import logging
from tqdm import tqdm
import h5py  # HDF5 포맷 저장을 위해 필요
import re

# 로그 시스템의 전체적인 규칙을 정하는 단계
# level=logging.INFO: "어느 정도 중요도의 로그까지 보여줄 것인가를 정함
# INFO로 설정하면 INFO, WARNING, ERROR, CRITICAL 수준의 로그는 출력되지만, DEBUG 수준의 로그는 무시
# format='%(levelname)s: %(message)s': 로그가 출력되는 모양을 결정
# %(levelname)s: 로그의 등급(예: INFO, ERROR)을 표시, %(message)s: 실제 기록하고자 하는 로그 내용을 표시

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# logger = logging.getLogger(__name__) 로그를 찍을 객체(Logger)를 생성하는 단계
# __name__: 현재 실행 중인 모듈(파일)의 이름을 나타내는 파이썬 특수 변수
logger = logging.getLogger(__name__)


class StrictSyncRoboticsDataset:
    def __init__(self, root_dir, target_hz=30.0, max_time_diff=0.05):
        self.root_dir = root_dir
        self.target_interval = 1.0 / target_hz # 0.03s
        self.max_time_diff = max_time_diff
        self.samples = []
        self.episode_ends = []

        # 이번에는 root_dir/episode, root_dir/episode_postech 둘 다 순회
        episode_folders = self._collect_episode_folders()

        current_idx = 0
        for ep_path, ep_name in episode_folders:
            kept_frames = self._process_episode_metadata(ep_path, ep_name)

            if kept_frames > 0:
                current_idx += kept_frames
                self.episode_ends.append(current_idx)

        logger.info(f"메타데이터 구축 완료! 총 {len(self.samples)}개의 동기화된 프레임이 식별되었습니다.")

    def _collect_episode_folders(self):
        """
        root_dir 아래의
          - episode/
          - episode_postech/
        두 폴더를 모두 확인하고,
        그 안의 실제 episode 폴더 경로를 모아 반환
        """
        candidate_parents = [
             os.path.join(self.root_dir, "episode"),
            # os.path.join(self.root_dir, "dock"),
            # os.path.join(self.root_dir, "valid"),
            # os.path.join(self.root_dir, "valid2"),
            # os.path.join(self.root_dir, "valid3"),
            # os.path.join(self.root_dir, "validation"),
        ]

        collected = []

        for parent in candidate_parents:
            if not os.path.exists(parent):
                logger.warning(f"상위 폴더가 없습니다. 스킵합니다: {parent}")
                continue
            
            # os.listdir(parent)를 통해 해당 상위 폴더 안에 있는 모든 파일/폴더를 가져옴
            # if os.path.isdir(...) 조건을 통해 파일은 무시하고 오직 폴더만 골라냄
            # sorted()를 사용하여 폴더명 순서대로
            subdirs = sorted(
                [
                    d for d in os.listdir(parent)
                    if os.path.isdir(os.path.join(parent, d))and d.lower().endswith("t")
                ],
                key=lambda x: int(re.search(r"record_(\d+)", x).group(1))
            )

            for d in subdirs:
                ep_path = os.path.join(parent, d)
                # 이름 충돌 방지를 위해 상위 폴더명까지 포함
                # os.path.basename: 주어진 경로 문자열에서 가장 마지막에 위치한 파일이나 폴더의 이름(이름 그 자체)만 잘라서 가져옴
                ep_name = f"{os.path.basename(parent)}/{d}"
                collected.append((ep_path, ep_name))

        logger.info(f"총 {len(collected)}개의 episode 폴더를 발견했습니다.")
        return collected

    def _normalize_timestamps(self, ts):
        ts = np.asarray(ts, dtype=np.float64)
        return ts / 1e9


    def _get_image_timestamps(self, img_dir):
        if not os.path.exists(img_dir):
            return np.array([]), []

        files = sorted([f for f in os.listdir(img_dir) if f.endswith('.jpg')])
        timestamps, valid_files = [], []

        for f in files:
            try:
                ts = float(f.replace('.jpg', '').split('_')[-1])
                timestamps.append(ts)
                valid_files.append(os.path.join(img_dir, f))
            except (ValueError, IndexError):
                continue

        return self._normalize_timestamps(np.array(timestamps)), valid_files

    def _process_episode_metadata(self, ep_path, ep_name):
        """실제 이미지를 로드하지 않고 경로와 센서 데이터만 매칭하여 리스트에 저장"""
        enc_path = os.path.join(ep_path, 'encoder.csv')

        camera_names = [
            "camera_orbbec-0",
            "camera_orbbec-2",
            "camera_orbbec-3",
            "camera_usb-0",
        ]

        if not os.path.exists(enc_path):
            logger.warning(f"❌ [{ep_name}] 스킵: encoder.csv 파일이 없습니다.")
            return 0

        df_enc = pd.read_csv(enc_path)
        df_enc['ts'] = self._normalize_timestamps(df_enc['ts'].values)

        required_cols = ['ts', 'vx', 'wz']
        for col in required_cols:
            if col not in df_enc.columns:
                logger.warning(f"❌ [{ep_name}] 스킵: encoder.csv에 '{col}' 컬럼이 없습니다.")
                return 0

        camera_data = {}

        for cam in camera_names:
            img_dir = os.path.join(ep_path, cam, 'frames')

            if not os.path.exists(img_dir):
                logger.warning(f"❌ [{ep_name}] 스킵: {cam}/frames 폴더가 없습니다.")
                return 0

            ts, files = self._get_image_timestamps(img_dir)

            if len(ts) == 0:
                logger.warning(f"❌ [{ep_name}] 스킵: {cam}/frames 이미지가 비어 있습니다.")
                return 0

            camera_data[cam] = {
                "ts": ts,
                "files": files,
            }

        min_t = max(
            [df_enc['ts'].min()] +
            [camera_data[cam]["ts"].min() for cam in camera_names]
        )

        max_t = min(
            [df_enc['ts'].max()] +
            [camera_data[cam]["ts"].max() for cam in camera_names]
        )
        
        print(f"\n===== {ep_name} =====")
        print(
            f"encoder=({df_enc['ts'].min():.3f}, "
            f"{df_enc['ts'].max():.3f})"
        )

        for cam in camera_names:
            print(
                f"{cam}=("
                f"{camera_data[cam]['ts'].min():.3f}, "
                f"{camera_data[cam]['ts'].max():.3f})"
            )

        print(
            f"common=({min_t:.3f}, {max_t:.3f}) "
            f"duration={max_t-min_t:.3f}s"
        )


        print(f"\n===== {ep_name} =====")

        for cam in camera_names:
            cam_min = camera_data[cam]['ts'].min()
            cam_max = camera_data[cam]['ts'].max()

            print(
                f"{cam:<20} "
                f"start_diff={cam_min - df_enc['ts'].min():8.3f}s "
                f"end_diff={cam_max - df_enc['ts'].max():8.3f}s "
                f"range=({cam_min:.3f}, {cam_max:.3f})"
            )

        print(
            f"common=({min_t:.3f}, {max_t:.3f}) "
            f"duration={max_t-min_t:.3f}s"
        )



        if min_t >= max_t:
            logger.warning(f"❌ [{ep_name}] 스킵: 유효한 공통 시간 구간이 없습니다.")
            return 0

        target_ts = np.arange(min_t, max_t, self.target_interval)

        interp_enc = interp1d(
            df_enc['ts'],
            df_enc[['vx', 'wz']].values,
            axis=0,
            kind='linear',
            fill_value="extrapolate"
        )

        synced_enc = interp_enc(target_ts)

        count = 0

        for i, t in enumerate(target_ts):
            sample = {
                'encoder': synced_enc[i].astype(np.float32),
                'image_paths': {}
            }

            valid = True

            for cam in camera_names:
                ts = camera_data[cam]["ts"]
                files = camera_data[cam]["files"]

                idx = np.abs(ts - t).argmin()

                if abs(ts[idx] - t) > self.max_time_diff:
                    valid = False
                    break

                sample['image_paths'][cam] = files[idx]

            if not valid:
                continue

            self.samples.append(sample)
            count += 1

        logger.info(f"[{ep_name}] 동기화 완료: {count} 프레임")
        return count



    def _load_image(self, img_path):
        img = cv2.imread(img_path)
        if img is None:
            return np.zeros((3, 240, 320), dtype=np.uint8)

        if img.shape[:2] != (240, 320):
            img = cv2.resize(img, (320, 240), interpolation=cv2.INTER_LINEAR)

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img_rgb.transpose(2, 0, 1)

    def __len__(self):
        return len(self.samples)

    def get_full_sample(self, idx):
        """파일 저장을 위해 한 프레임의 모든 데이터를 로드"""
        s = self.samples[idx]

        enc = s['encoder']
        images = {}

        for cam, img_path in s['image_paths'].items():
            images[cam] = self._load_image(img_path)

        return enc, images


if __name__ == "__main__":
    import argparse

    # argparse 라이브러리 - 터미널에서 명령어를 입력할 때 필요한 정보(인자)를 받아오는 표준 방식
    # 인자를 처리할 parser(분석기) 객체를 생성 description은 --help를 입력했을 때 나타나는 설명문
    parser = argparse.ArgumentParser(description="Preprocess raw episode data into HDF5 format")

    # parser.add_argument("--data_root", ...): required=True: 이 인자를 입력하지 않으면 에러를 발생, 프로그램 종료(필수 입력값)
    parser.add_argument("--data_root", type=str, required=True,
                        help="Root directory containing validation/ folder with episode data")
    parser.add_argument("--save_path", type=str, required=True,
                        help="Output HDF5 file path (e.g. ./validation_dataset.h5)")
    # parser.parse_args(): 사용자가 터미널에 입력한 명령어를 파싱(분석)하여 객체에 저장
    cli_args = parser.parse_args()

    DATA_ROOT_DIR = cli_args.data_root
    SAVE_PATH = cli_args.save_path

    dataset = StrictSyncRoboticsDataset(root_dir=DATA_ROOT_DIR)
    total_samples = len(dataset)
    
    print(f"\n데이터 저장을 시작합니다: {SAVE_PATH}")

    CAMERA_NAMES = [
        "camera_orbbec-0",
        "camera_orbbec-2",
        "camera_orbbec-3",
        "camera_usb-0",
    ]

    with h5py.File(SAVE_PATH, 'w') as f:
        dset_enc = f.create_dataset("encoder", (total_samples, 2), dtype='f4')
        image_dsets = {}

        for cam in CAMERA_NAMES:
            h5_name = cam.replace("-", "_")

            image_dsets[cam] = f.create_dataset(
                h5_name,
                (total_samples, 3, 240, 320),
                dtype='u1',
                compression="gzip",
                chunks=(1, 3, 240, 320)
            )

        f.create_dataset(
            "episode_ends",
            data=np.array(dataset.episode_ends, dtype='i8')
        )

        for i in tqdm(range(total_samples), desc="Disk Writing"):
            enc, images = dataset.get_full_sample(i)

            dset_enc[i] = enc

            for cam in CAMERA_NAMES:
                image_dsets[cam][i] = images[cam]

    print(f"\n모든 데이터가 안전하게 저장되었습니다! (최종 프레임: {total_samples})")
