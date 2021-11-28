"""resizer module handle image resizing"""

import albumentations as A
import cv2
import numpy as np


class Resizer:
    """Resize images"""

    def __init__(self, image_size, resize_mode, resize_only_if_bigger):
        self.image_size = image_size
        self.resize_mode = resize_mode
        self.resize_only_if_bigger = resize_only_if_bigger

        if resize_mode not in ["no", "keep_ratio", "center_crop", "border"]:
            raise Exception(f"Invalid option for resize_mode: {resize_mode}")

        if resize_mode == "keep_ratio":
            self.resize_tfm = A.SmallestMaxSize(image_size, interpolation=cv2.INTER_LANCZOS4)
        elif resize_mode == "center_crop":
            self.resize_tfm = A.Compose(
                [A.SmallestMaxSize(image_size, interpolation=cv2.INTER_LANCZOS4), A.CenterCrop(image_size, image_size),]
            )
        elif resize_mode == "border":
            self.resize_tfm = A.Compose(
                [
                    A.LongestMaxSize(image_size, interpolation=cv2.INTER_LANCZOS4),
                    A.PadIfNeeded(image_size, image_size, border_mode=cv2.BORDER_CONSTANT, value=[255, 255, 255],),
                ]
            )
        elif resize_mode == "no":
            pass

    def __call__(self, img_stream):
        try:
            img = cv2.imdecode(np.frombuffer(img_stream.read(), np.uint8), cv2.IMREAD_UNCHANGED)
            if img is None:
                raise Exception("Image decoding error")
            if len(img.shape) == 3 and img.shape[-1] == 4:
                # alpha matting with white background
                alpha = img[:, :, 3, np.newaxis]
                img = alpha / 255 * img[..., :3] + 255 - alpha
                img = np.rint(img.clip(min=0, max=255)).astype(np.uint8)
            original_height, original_width = img.shape[:2]

            # resizing in following conditions
            if self.resize_mode != "no" and (
                not self.resize_only_if_bigger
                or (
                    self.resize_mode in ["keep_ratio", "center_crop"]
                    # smallest side contained to image_size (largest cropped)
                    and min(img.shape[:2]) > self.image_size
                )
                or (
                    self.resize_mode == "border"
                    # largest side contained to image_size
                    and max(img.shape[:2]) > self.image_size
                )
            ):
                img = self.resize_tfm(image=img)["image"]

            height, width = img.shape[:2]
            img_str = cv2.imencode(".jpg", img)[1].tobytes()
            del img
            return img_str, width, height, original_width, original_height, None

        except Exception as err:  # pylint: disable=broad-except
            return None, None, None, None, None, str(err)
