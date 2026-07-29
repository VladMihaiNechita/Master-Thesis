from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import torch

# This approach requires 94.87 of CPU memory
class RealVideoGenerator:
    def __init__(self, image_size, video_path="datasets/video/Walking_in_Amsterdam_640x360_30fps.mp4"):
        self.image_size = image_size
        self.video_path = Path(video_path)
        if not self.video_path.is_absolute():
            self.video_path = Path(__file__).resolve().parents[2] / self.video_path

        self.image_channels = 3
        self.frame_width = 640
        self.frame_height = 360
        self.dtype = torch.float32
        self.device = 'cuda'

        video = self.open_video()
        self.frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        video.release()

        # Decode the video once; its RGB uint8 cache is reused for every batch.
        self.frames = np.empty((self.frame_count, self.frame_height, self.frame_width,
                                self.image_channels), dtype=np.uint8)
        self.decode_video()

        self.crop_executor = ThreadPoolExecutor(max_workers=2)
        self.prefetch_executor = ThreadPoolExecutor(max_workers=1)
        self.prefetched_batch = None
        self.prefetched_batch_index = None
        self.closed = False

    def open_video(self):
        return cv2.VideoCapture(str(self.video_path), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1])

    def decode_range(self, start_frame, end_frame):
        video = self.open_video()
        video.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        try:
            for frame_index in range(start_frame, end_frame):
                success, frame = video.read()
                if not success:
                    raise RuntimeError(f"Could only decode {frame_index} frames from {self.video_path}.")
                self.frames[frame_index] = frame[:, :, ::-1]
        finally:
            video.release()

    def decode_video(self):
        decoder_count = min(32, self.frame_count)
        with ThreadPoolExecutor(max_workers=decoder_count) as executor:
            futures = []
            for decoder_index in range(decoder_count):
                start_frame = decoder_index * self.frame_count // decoder_count
                end_frame = (decoder_index + 1) * self.frame_count // decoder_count
                futures.append(executor.submit(self.decode_range, start_frame, end_frame))
            for future in futures:
                future.result()

    def copy_crops(self, images, frame_indices, top, left, start_index, end_index):
        for index in range(start_index, end_index):
            y = top[index]
            x = left[index]
            images[index] = self.frames[frame_indices[index], y:y + self.image_size, x:x + self.image_size]

    def read_batch(self, batch_size, batch_index):
        generator = torch.Generator().manual_seed(1729 + batch_index)

        # Stage 1: pick one random frame from each interval of the video.
        interval_starts = torch.arange(batch_size) * self.frame_count // batch_size
        interval_ends = torch.arange(1, batch_size + 1) * self.frame_count // batch_size
        frame_indices = interval_starts + (torch.rand(batch_size, generator=generator) * (interval_ends - interval_starts)).to(torch.int64)
        frame_indices = frame_indices.numpy()

        # Stage 2: copy random square crops into pinned memory using two workers.
        top = torch.randint(0, self.frame_height - self.image_size + 1, (batch_size,), generator=generator).numpy()
        left = torch.randint(0, self.frame_width - self.image_size + 1, (batch_size,), generator=generator).numpy()
        images = torch.empty(batch_size, self.image_size, self.image_size, self.image_channels, dtype=torch.uint8, pin_memory=True)
        images_numpy = images.numpy()

        middle = batch_size // 2
        first = self.crop_executor.submit(self.copy_crops, images_numpy, frame_indices, top, left, 0, middle)
        second = self.crop_executor.submit(self.copy_crops, images_numpy, frame_indices, top, left, middle, batch_size)
        first.result()
        second.result()

        return images

    def generate(self, batch_size, batch_index):
        if self.prefetched_batch_index == batch_index:
            images = self.prefetched_batch.result()
        else:
            images = self.read_batch(batch_size, batch_index)

        # Prepare the next deterministic batch while the GPU trains on this one.
        self.prefetched_batch_index = batch_index + 1
        self.prefetched_batch = self.prefetch_executor.submit(
            self.read_batch, batch_size, self.prefetched_batch_index)

        # Stage 3: match the other generators: [batch, channels, height, width], float32, [0, 1].
        images = images.permute(0, 3, 1, 2)
        images = images.to(device=self.device, dtype=self.dtype, non_blocking=True) / 255.0

        return images

    def close(self):
        if self.closed:
            return
        if self.prefetched_batch is not None:
            self.prefetched_batch.cancel()
        self.prefetch_executor.shutdown()
        self.crop_executor.shutdown()
        self.closed = True

    def __del__(self):
        if hasattr(self, "closed"):
            self.close()


_video_generator = None
_video_generator_key = None


def generate_from_real_video(batch_size, image_size, batch_index, video_path="datasets/video/Walking_in_Amsterdam_640x360_30fps.mp4"):
    global _video_generator
    global _video_generator_key

    video_generator_key = (batch_size, image_size, str(video_path))
    if _video_generator is None or _video_generator_key != video_generator_key:
        if _video_generator is not None:
            _video_generator.close()
        _video_generator = RealVideoGenerator(image_size=image_size, video_path=video_path)
        _video_generator_key = video_generator_key

    return _video_generator.generate(batch_size, batch_index)
