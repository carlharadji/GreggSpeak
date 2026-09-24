from app.pipeline.runner import PageInput, recognize_pages


class RecognitionService:
    def __init__(self, use_uniform_crop=False):
        self.use_uniform_crop = use_uniform_crop

    def is_available(self):
        return True

    def recognize_pages(self, pages, laptop_compatible=True, apply_fixed_crop=None):
        page_inputs = [
            PageInput(
                source_name=f"lcd_page_{index}",
                image=page,
                image_path=f"lcd_page_{index}",
                apply_fixed_crop=apply_fixed_crop,
            )
            for index, page in enumerate(pages, start=1)
        ]
        return recognize_pages(
            page_inputs,
            use_uniform_crop=self.use_uniform_crop,
            laptop_compatible=laptop_compatible,
        )
