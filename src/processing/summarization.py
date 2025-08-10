from transformers import BlipProcessor, BlipForConditionalGeneration
import fitz
from PIL import Image
import io

def _normalize_text(s: str) -> str:
    return " ".join(s.split()).strip().lower()

class BLIPSummarizer:
    def __init__(self, model_name: str, device: str = "cpu"):
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(model_name).to(device)
        self.device = device

    def _generate_text(self, inputs, max_new_tokens=200,
                       num_beams=4, no_repeat_ngram_size=3,
                       repetition_penalty=1.2, temperature=1.0, top_p=0.95):
        gen_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            no_repeat_ngram_size=no_repeat_ngram_size,
            repetition_penalty=repetition_penalty,
            temperature=temperature,
            top_p=top_p,
            early_stopping=True,
        )
        return self.processor.decode(gen_ids[0], skip_special_tokens=True).strip()

    def summarize_financial_page(self, image, prompt: str = None) -> str:
        # Default concise instruction if not provided
        if prompt is None:
            prompt = (
                "Summarize the financial data on this page: tables, numeric values, and key takeaways. "
                "Output a short clear summary."
            )

        # Prepare inputs with image + prompt
        inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.device)
        generated_text = self._generate_text(inputs, max_new_tokens=200)

        # Detect prompt echoes or poor generation
        norm_prompt = _normalize_text(prompt)
        norm_gen = _normalize_text(generated_text or "")

        looks_like_prompt_echo = (
            not norm_gen or
            (norm_prompt in norm_gen) or
            norm_gen.split()[:6] == norm_prompt.split()[:6]  # starts with same first words
        )

        if looks_like_prompt_echo:
            # fallback: image-only captioning (shorter); often avoids echo/repetition
            cap_inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            caption = self._generate_text(cap_inputs, max_new_tokens=80, num_beams=5, no_repeat_ngram_size=2)
            return caption or "No summary generated"

        return generated_text

    @staticmethod
    def pdf_page_to_image(pdf_path: str, page_number: int, dpi: int = 300, crop_box: tuple = None) -> Image.Image:
        """
        Utility: render a PDF page to PIL.Image.
        crop_box: optional (x0, y0, x1, y1) in PDF points (same units fitz uses). If None, full page.
        """
        doc = fitz.open(pdf_path)
        page = doc.load_page(page_number - 1)
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        clip = fitz.Rect(*crop_box) if crop_box else None
        pix = page.get_pixmap(matrix=mat, clip=clip)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        doc.close()
        return img
