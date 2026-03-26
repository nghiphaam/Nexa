"""Curriculum loader for multimodal data."""
from __future__ import annotations

import random

import numpy as np
from PIL import Image


class CurriculumDataLoader:
    def __init__(self, data, curriculum_ratio=0.9):
        self.data = self._sort_by_complexity(data)
        self.curriculum_ratio = curriculum_ratio

    def _get_image_value(self, sample):
        if 'image' in sample:
            return sample['image']
        if 'images' in sample:
            images = sample['images']
            if isinstance(images, (list, tuple)):
                return images[0] if images else None
            return images
        return None

    def _sort_by_complexity(self, data):
        complexities = []
        for sample in data:
            image_value = self._get_image_value(sample)
            complexity = self._estimate_complexity(image_value)
            complexities.append((complexity, sample))
        complexities.sort(key=lambda x: x[0])
        return [s for _, s in complexities]

    def _estimate_complexity(self, image_value):
        if image_value is None:
            return 0.0
        try:
            if isinstance(image_value, str):
                img = Image.open(image_value).convert('RGB')
                img_array = np.array(img)
            elif hasattr(image_value, 'detach'):
                img_array = image_value.detach().cpu().float().numpy()
            else:
                img_array = np.array(image_value)
            return float(np.asarray(img_array).std())
        except Exception:
            return 0.5

    def get_epoch_data(self, epoch):
        n = len(self.data)
        if n == 0:
            return []

        if epoch < 2:
            curriculum_data = self.data[: max(1, n // 2)]
        elif epoch < 4:
            curriculum_data = self.data[n // 4: max(n // 4 + 1, 3 * n // 4)]
        else:
            curriculum_data = self.data

        num_curriculum = int(len(curriculum_data) * self.curriculum_ratio)
        num_random = len(curriculum_data) - num_curriculum

        curriculum_samples = curriculum_data[:num_curriculum]
        random_samples = random.sample(self.data, min(num_random, len(self.data))) if num_random > 0 else []

        mixed_data = curriculum_samples + random_samples
        random.shuffle(mixed_data)
        return mixed_data
