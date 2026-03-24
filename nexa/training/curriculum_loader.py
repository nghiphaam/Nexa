"""Curriculum loader for multimodal data."""
import random
from PIL import Image
import numpy as np


class CurriculumDataLoader:
    def __init__(self, data, curriculum_ratio=0.9):
        self.data = self._sort_by_complexity(data)
        self.curriculum_ratio = curriculum_ratio

    def _sort_by_complexity(self, data):
        complexities = []
        for sample in data:
            complexity = self._estimate_complexity(sample['image'])
            complexities.append((complexity, sample))
        complexities.sort(key=lambda x: x[0])
        return [s for _, s in complexities]

    def _estimate_complexity(self, image_path):
        try:
            img = Image.open(image_path).convert('RGB')
            img_array = np.array(img)
            return img_array.std()
        except Exception:
            return 0.5  # fallback

    def get_epoch_data(self, epoch):
        n = len(self.data)

        # Get curriculum subset
        if epoch < 2:
            curriculum_data = self.data[:n//2]
        elif epoch < 4:
            curriculum_data = self.data[n//4:3*n//4]
        else:
            curriculum_data = self.data

        # Mix 90% curriculum + 10% random full data
        num_curriculum = int(len(curriculum_data) * self.curriculum_ratio)
        num_random = len(curriculum_data) - num_curriculum

        curriculum_samples = curriculum_data[:num_curriculum]
        random_samples = random.sample(self.data, min(num_random, len(self.data)))

        mixed_data = curriculum_samples + random_samples
        random.shuffle(mixed_data)

        return mixed_data
